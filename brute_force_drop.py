import argparse
import random
import os
import copy
from tqdm import trange
from typing import List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    import wandb

    has_wandb = True
except ModuleNotFoundError:
    has_wandb = False

from src.data_utils import get_data
from src.common_utils import fix_seed
from src.model_utils import (
    get_layers,
    get_attn_layer_name,
    get_mlp_layer_name,
    make_dummy_forward,
    dummy_initialize,
    restore_forward,
)
from src.metrics import compute_perplexity, compute_kl_div


def get_layer_drop_config(removed_state) -> List[str]:
    num_blocks = len(removed_state["attn"])
    drop_config = ["none" * num_blocks]
    for i in range(num_blocks):
        if removed_state["attn"][i] and removed_state["mlp"][i]:
            drop_config[i] = "attn+mlp"
        elif removed_state["attn"][i]:
            drop_config[i] = "attn"
        elif removed_state["mlp"][i]:
            drop_config[i] = "mlp"
    return drop_config


def load_states(model, layers, removed_state, drop_two_consecutive):
    removed_state = copy.deepcopy(removed_state)
    if drop_two_consecutive:  # decompress: duplicate every entry
        removed_state["attn"] = [removed_state["attn"][i // 2] for i in range(2 * len(removed_state["attn"]))]
        removed_state["mlp"] = [removed_state["mlp"][i // 2] for i in range(2 * len(removed_state["mlp"]))]

    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model))
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            else:
                restore_forward(subblock)


def compute_fitness(model, data, fitness_fn, invert_fitness, target_logits: Optional[torch.Tensor] = None) -> float:
    sign = 1
    if invert_fitness:
        sign = -1

    if fitness_fn == "ppl":
        return sign * compute_perplexity(model, data)
    else:
        return sign * compute_kl_div(model, data, target_logits)


def parse_args():
    parser = argparse.ArgumentParser(description="Layer dropping.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", required=True, type=int, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--calibration_streaming", action="store_true", help="Whether to load calibration data in streaming mode."
    )
    parser.add_argument("--sequence_length", default=None, type=int, help="Length of sequences.")
    parser.add_argument("--fitness_fn", choices=["ppl", "kl"], default="kl", help="Fitness function.")

    # Sparsification params
    parser.add_argument("--sparsity", type=float, required=True, help="Fraction of layers to drop.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")

    parser.add_argument("--drop_entire_block", action="store_true", help="Whether to drop entire block (attn+mlp).")
    parser.add_argument(
        "--drop_two_consecutive",
        action="store_true",
        help="If set can only drop two consecutive blocks together (first and second, third and fourth,...). Can only be set when entire blocks are dropped.",
    )
    # Save params
    parser.add_argument("--save_dir", type=str, help="where to save sparse model.")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert args.drop_entire_block, "Brute force only implemented for entire block"
    # Get device and dtype
    assert torch.cuda.is_available()
    device = f"cuda"
    dtype = getattr(torch, args.dtype)
    # Fix seed
    fix_seed(args.seed)
    # Init W&B logger
    if args.log_wandb:
        assert has_wandb, "`wandb` not installed, try pip install `wandb`"
        wandb.init(config=args)
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map="auto",
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.config.use_cache = False  # do not use cache
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, use_fast=args.use_fast_tokenizer, trust_remote_code=True
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.calibration_sequence_length,
        tokenizer,
        train=True,
        streaming=args.calibration_streaming,
    )

    layers = get_layers(model)
    blocks_to_remove = int(args.sparsity * len(layers))
    print(f"Removing {blocks_to_remove} blocks")
    total_blocks = len(layers)
    if args.drop_two_consecutive:
        assert total_blocks % 2 == 0 and blocks_to_remove % 2 == 0, "Total blocks and removed blocks must be even"
        total_blocks = total_blocks // 2  # view two consecutive blocks as one block
        blocks_to_remove = blocks_to_remove // 2

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model)))

    all_candidates = []

    num_tested = 0
    for i in range(1 << total_blocks):  # iterate over all bitstrings of length total_blocks
        if i.bit_count() != blocks_to_remove:
            continue

        cand = [False] * total_blocks
        for j in range(total_blocks):
            if i & (1 << j):
                cand[j] = True
        num_tested += 1
        print(num_tested)
        removed_state = {"attn": cand, "mlp": cand}
        load_states(model, layers, removed_state, args.drop_two_consecutive)
        fitness = compute_perplexity(model, calibration_data)
        print(fitness, removed_state)
        all_candidates.append((fitness, removed_state))
    all_candidates = sorted(all_candidates, key=lambda x: x[0])
    count = 0
    for loss, config in all_candidates:
        print(count, loss, config)
        count += 1
    count = 0


if __name__ == "__main__":
    main()
