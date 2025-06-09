import argparse
import time
import os

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM

from src import dist_utils
from src.data_utils import get_data
from src.error_estimator import ErrorEstimator


def parse_args():
    parser = argparse.ArgumentParser(description="Later dropping.")
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="The name or path to the model being pruned",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="The name or path to the tokenizer. By default use model tokenizer.",
    )
    parser.add_argument(
        "--target_modules",
        type=str,
        required=True,
        help="Regex for modules to prune",
    )
    parser.add_argument(
        "--pre_block_modules",
        nargs="+",
        type=str,
        required=True,
        help="Names of modules before transformer blocks",
    )
    parser.add_argument(
        "--block_modules",
        type=str,
        required=True,
        help="Name of transformer modules",
    )
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_samples", default=64, type=int, help="Number of samples for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", default=None, type=int, help="Length of calibration sequences."
    )
    # Сompression params
    parser.add_argument(
        "--compressed_weights_path", type=str, required=True, help="Path to sparse or quantized weights"
    )
    parser.add_argument("--group_by_numel", action="store_true", help="whether to group l2 error by number of elements")
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--low_cpu_mem_usage", action="store_true", help="whether to load model with the use of `low_cpu_mem_usage`"
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation for both teacher and student models: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    # Save params
    parser.add_argument(
        "--configuration_name", type=str, default="final_configuration.txt", help="Name of final configuration"
    )
    parser.add_argument("--cpu_offload_modules", action="store_true", help="whether to offload modules to CPU.")
    parser.add_argument("--cpu_offload_activations", action="store_true", help="whether to offload activations to CPU.")
    parser.add_argument("--verbose", action="store_true", help="whether to log progress.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # Distributed init
    if dist.is_available():
        dist.init_process_group(backend="nccl", init_method="env://")
    world_size = dist_utils.get_world_size()
    rank = dist_utils.get_rank()
    # init device
    device = f"cuda:{rank}"
    if args.dtype != "auto":
        args.dtype = getattr(torch, args.dtype)
    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=args.dtype,
        low_cpu_mem_usage=args.low_cpu_mem_usage,
        attn_implementation=args.attn_implementation,
    )
    if not args.cpu_offload_modules:
        model = model.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name or args.model_name_or_path, use_fast=False)
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data, args.calibration_samples, args.calibration_sequence_length, tokenizer, train=True
    )
    # take slice (if running on multiple workers)
    if dist_utils.is_dist_available_and_initialized():
        num_seq_per_rank = len(calibration_data) // world_size
        calibration_data = calibration_data[rank * num_seq_per_rank : (rank + 1) * num_seq_per_rank]
    calibration_data = [([], {"input_ids": input_ids}) for input_ids in calibration_data]
    dist.barrier()
    # Pruner
    error_estimator = ErrorEstimator(
        model,
        calibration_data,
        target_modules=args.target_modules,
        pre_block_modules=args.pre_block_modules,
        block_modules=args.block_modules,
        compressed_weights_path=args.compressed_weights_path,
        device=device,
        cpu_offload_modules=args.cpu_offload_modules,
        cpu_offload_activations=args.cpu_offload_activations,
        verbose=args.verbose,
    )
    dist.barrier()
    t1 = time.perf_counter()
    errors = error_estimator.estimate(args.group_by_numel)
    t2 = time.perf_counter()
    dist_utils.print_on_main(f"Error estimate took {(t2 - t1)} s.")
    if dist_utils.is_main():
        os.makedirs(os.path.join(args.compressed_weights_path), exist_ok=True)
        torch.save(errors, os.path.join(args.compressed_weights_path, "errors.pth"))


if __name__ == "__main__":
    main()
