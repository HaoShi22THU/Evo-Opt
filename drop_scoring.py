import argparse
import tqdm
from tqdm import trange

import torch
import torch.nn.functional as F
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


@torch.no_grad()
def compute_cosine_similarity(X_before, X_after, device):
    cosine_sim = F.cosine_similarity(X_before.to(device), X_after.to(device), dim=1)
    average_cosine_sim = cosine_sim.mean()
    return average_cosine_sim.item()


@torch.no_grad()
def compute_l2_error(X_before, X_after, device):
    l2_error = torch.norm(X_before.to(device) - X_after.to(device), p=2, dim=1).mean()
    return l2_error.item()


@torch.no_grad()
def compute_l2_error_normalized(X_before, X_after, device):
    X_before = F.normalize(X_before, p=2, dim=1)
    X_after = F.normalize(X_after, p=2, dim=1)
    l2_error = torch.norm(X_before.to(device) - X_after.to(device), p=2, dim=1).mean()
    return l2_error.item()


@torch.no_grad()
def compute_norm_ratio(X_before, X_after, device):
    print(X_before.shape, X_after.shape)
    X_before = X_before.float().to(device)
    X_after = X_after.float().to(device)
    with torch.no_grad():
        score = ((X_after - X_before).norm(dim=1) / X_after.norm(dim=1)).mean()
        return float(score)


@torch.no_grad()
def get_embeddings(model, data):
    layer_to_embs = {}
    for j in range(len(data)):
        print(f"{j}/{len(data)}")
        outputs = model(data[j].to(model.device), output_hidden_states=True)
        hidden_states = outputs.hidden_states
        for i, emb in enumerate(hidden_states):
            if i in layer_to_embs:
                layer_to_embs[i] += list(emb.cpu().detach()[0])
            else:
                layer_to_embs[i] = list(emb.cpu().detach()[0])
        del outputs
    for i in range(len(hidden_states)):
        layer_to_embs[i] = torch.stack(layer_to_embs[i])
    return layer_to_embs


def parse_args():
    parser = argparse.ArgumentParser()
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
    # Data params
    parser.add_argument(
        "--calibration_data",
        type=str,
        required=True,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", type=int, required=True, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", type=int, required=True, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=["fineweb_edu", "wikitext2", "c4"],
        help="Datasets used for evaluation",
    )
    parser.add_argument("--no_eval", action="store_true", help="Whether to skip evaluation")
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument(
        "--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation (not used for wiki2/c4)."
    )
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Scoring params
    parser.add_argument(
        "--scoring_method",
        type=str,
        choices=[
            "cosine_similarity",
            "perplexity",
            "window_cosine_similarity",
            "norm_ratio",
            "kl_div",
            "l2",
            "l2_normalized",
        ],
        help="Scoring method for layer dropping.",
    )
    parser.add_argument("--sparsities", nargs="+", type=float, help="Sparsities to evaluate")
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
    print(model.config.model_type)
    print(model)
    model.config.use_cache = False  # do not use cache
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name or args.model_name_or_path, use_fast=args.use_fast_tokenizer
    )
    # Load calibration data
    args.calibration_sequence_length = args.calibration_sequence_length or model.config.max_position_embeddings
    calibration_data = get_data(
        args.calibration_data,
        args.calibration_tokens,
        args.calibration_sequence_length,
        tokenizer,
        train=True,
    )
    # Load evaluation data
    args.sequence_length = args.eval_sequence_length or model.config.max_position_embeddings
    eval_datasets = []
    for eval_dataset_name in args.eval_datasets:
        eval_datasets.append(
            get_data(
                eval_dataset_name,
                args.eval_tokens,  # ignored for WikiText2 and C4
                args.eval_sequence_length,
                tokenizer,
                train=False,
            )
        )

    target_logits = []
    if args.scoring_method == "kl_div":
        # Compute target logits (calibration)
        for i in trange(0, len(calibration_data), desc="Computing target logits (calib)", leave=False):
            with torch.no_grad():
                target_logits.append(model(calibration_data[i].to(device)).logits.cpu())

    layers = get_layers(model)
    total_blocks = len(layers)
    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model)))
        dummy_initialize(layer)

    if args.scoring_method == "window_cosine_similarity":
        embs_dict = get_embeddings(model, calibration_data)

        for sparsity in args.sparsities:
            num_dropped = int(sparsity * total_blocks)
            scores = []

            for start in range(0, total_blocks + 1 - num_dropped):
                score = -compute_cosine_similarity(
                    embs_dict[start], embs_dict[start + num_dropped], device
                )  # negative because scores measures importance
                scores.append((score, start))
                print(f"start {start}, score {score}")

            scores.sort()
            best_start = scores[0][1]
            for ind in range(num_dropped):  # drop for evaluations of best start
                make_dummy_forward(layers[best_start + ind], "attn+mlp")

            print(f"Best start for {num_dropped} blocks: {best_start}")

            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")

            print("=" * 20)

            for layer in layers:  # restore all blocks
                restore_forward(layer)

    else:  # methods that rank each block separately, yielding a remove order
        if args.scoring_method == "perplexity":
            scores = []
            for layer_id, layer in enumerate(layers):
                make_dummy_forward(layer, "attn+mlp")
                ppl = compute_perplexity(model, calibration_data)
                restore_forward(layer)
                print(f"Perplexity for layer {layer_id} dropped: {ppl:.2f}")
                scores.append((ppl, layer_id))
            scores.sort()
            remove_order = [layer_id for _, layer_id in scores]
        elif args.scoring_method == "kl_div":
            scores = []
            for layer_id, layer in enumerate(layers):
                make_dummy_forward(layer, "attn+mlp")
                kl_div = compute_kl_div(model, calibration_data, target_logits)
                restore_forward(layer)
                print(f"KL Divergence for layer {layer_id} dropped: {kl_div:.2f}")
                scores.append((kl_div, layer_id))
            scores.sort()
            remove_order = [layer_id for _, layer_id in scores]
        else:  # methods that compute a score for each block (without evaluating entire model)
            embs_dict = get_embeddings(model, calibration_data)

            scores = []

            for layer_id in range(max(embs_dict.keys())):
                if args.scoring_method == "cosine_similarity":
                    scores.append(
                        (-compute_cosine_similarity(embs_dict[layer_id], embs_dict[layer_id + 1], device), layer_id)
                    )  # negative because scores measures importance
                elif args.scoring_method == "norm_ratio":
                    scores.append((compute_norm_ratio(embs_dict[layer_id], embs_dict[layer_id + 1], device), layer_id))
                elif args.scoring_method == "l2":
                    scores.append((compute_l2_error(embs_dict[layer_id], embs_dict[layer_id + 1], device), layer_id))
                elif args.scoring_method == "l2_normalized":
                    scores.append(
                        (compute_l2_error_normalized(embs_dict[layer_id], embs_dict[layer_id + 1], device), layer_id)
                    )
                else:
                    raise NotImplementedError(f"Scoring method {args.scoring_method} not implemented.")

            scores.sort()
            remove_order = [layer_id for _, layer_id in scores]

        print(f"Remove order:{remove_order}")
        print(args.sparsities)

        for sparsity in args.sparsities:
            num_dropped = int(sparsity * total_blocks)

            for layer_id in remove_order[:num_dropped]:
                make_dummy_forward(layers[layer_id], "attn+mlp")

            print(f"Evaluating {num_dropped} blocks dropped...")
            for eval_dataset_name, eval_dataset in zip(args.eval_datasets, eval_datasets):
                ppl_eval = compute_perplexity(model, eval_dataset)
                print(f"{eval_dataset_name}: {ppl_eval:.2f}")
            print("=" * 20)

            for layer_id in remove_order[:num_dropped]:
                restore_forward(layers[layer_id])


if __name__ == "__main__":
    main()
