import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import argparse
import random
import os
import copy
import numpy as np
from tqdm import trange
from typing import List, Optional

import clip
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


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
# from src.metrics import compute_perplexity, compute_kl_div


### Janus Pro
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image


# @torch.no_grad()
# def compute_perplexity(model, data, batch_size: int = 1):
#     num_samples = len(data)
#     device = next(model.parameters()).device
#     # Running estimate of negative log-likelihood
#     nll_running = 0
#     # Number of tokens processed to far
#     tokens_processed = 0

#     cfg_weight = 5
#     temperature = 1.0
#     # Loop through each batch
#     for i in trange(0, num_samples, batch_size, desc="Computing perplexity", leave=False):
#         j = min(i + batch_size, num_samples)
#         inputs = data[:,:,i:j].to(device)
#         # Forward pass through the model
#         inputs = inputs.permute(0, 2, 1)

#         outputs = model.language_model.model(inputs_embeds=inputs, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
#         hidden_states = outputs.last_hidden_state
        
#         logits = model.gen_head(hidden_states[:, -1, :])
#         logit_cond = logits[0::2, :]
#         logit_uncond = logits[1::2, :]
        
#         logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
#         lm_logits = torch.softmax(logits / temperature, dim=-1)

#         # lm_logits = model(inputs).logits
#         # Shift logits and labels for next token prediction
#         shift_logits = lm_logits[:, :-1].contiguous()
#         shift_labels = inputs[:, 1:]
#         # Compute loss
#         loss = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))
#         # Calculate negative log likelihood
#         a = shift_labels.numel() / (tokens_processed + shift_labels.numel())
#         b = tokens_processed / (tokens_processed + shift_labels.numel())
#         nll_running = a * loss + b * nll_running
#         # Update number of processed tokens
#         tokens_processed += shift_labels.numel()
#     # Compute perplexity
#     ppl = nll_running.exp().item()
#     return ppl

# def calculate_ppl(model, generated_tokens, logits_history):
#     total_nll = 0.0
#     num_tokens = generated_tokens.shape[1]  # image_token_num_per_image
    
#     for i in range(num_tokens):
#         # 获取当前token的真实ID和对应logits
#         current_token = generated_tokens[:, i]  # shape: [parallel_size]
#         current_logits = logits_history[i]      # shape: [parallel_size, vocab_size]
#         # print(current_logits.dtype)
#         # 计算负对数似然
#         probs = torch.softmax(current_logits, dim=-1)
#         nll = -torch.log(probs.gather(1, current_token.unsqueeze(1).long())).mean()
#         total_nll += nll.item()
    
#     return torch.exp(torch.tensor(total_nll / num_tokens)).item()

# @torch.no_grad()
# def compute_perplexity(model, prompt, data, batch_size: int = 1):

#     device = next(model.parameters()).device
#     parallel_size = 1
#     image_token_num_per_image = 576
#     cfg_weight = 5
#     temperature = 1.0
    
#     inputs_embeds = prompt.to(device)
#     generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long).cuda()

#     logits_history = []
#     for i in range(576):
#         outputs = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
#         hidden_states = outputs.last_hidden_state
        
#         logits = model.gen_head(hidden_states[:, -1, :])
#         logit_cond = logits[0::2, :]
#         logit_uncond = logits[1::2, :]
        
#         logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
#         logits_history.append(logits)
#         probs = torch.softmax(logits / temperature, dim=-1)

#         next_token = torch.multinomial(probs, num_samples=1)
#         generated_tokens[:, i] = next_token.squeeze(dim=-1)

#         next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
#         img_embeds = model.prepare_gen_img_embeds(next_token)
        
#         inputs_embeds = img_embeds.unsqueeze(dim=1)
#     ppl = calculate_ppl(model, generated_tokens, logits_history)
#     # ppl = random.randint(0, 100)
#     return ppl

@torch.no_grad()
def compute_clip(model, model_full, clip_model, clip_preprocess, prompt):

    device = next(model.parameters()).device
    parallel_size = 1
    image_token_num_per_image = 576
    cfg_weight = 5
    temperature = 1.0

    img_size = 384
    patch_size = 16

    inputs_embeds = prompt.to(device)
    # print("inputs_embeds shape:", inputs_embeds.shape)

    generated_tokens = torch.zeros((parallel_size, image_token_num_per_image), dtype=torch.long).cuda()
    store_tokens = torch.zeros((parallel_size*2, 4096, image_token_num_per_image + 50), dtype=torch.bfloat16).cuda()
    store_tokens[:,:, :50] = inputs_embeds.permute(0, 2, 1)

    for i in range(image_token_num_per_image):
        if i <= image_token_num_per_image // 2:
            outputs = model_full.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = model_full.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = model_full.prepare_gen_img_embeds(next_token)
            
            inputs_embeds = img_embeds.unsqueeze(dim=1)
        else:
            outputs = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=outputs.past_key_values if i != 0 else None)
            hidden_states = outputs.last_hidden_state
            
            logits = model.gen_head(hidden_states[:, -1, :])
            logit_cond = logits[0::2, :]
            logit_uncond = logits[1::2, :]
            
            logits = logit_uncond + cfg_weight * (logit_cond-logit_uncond)
            probs = torch.softmax(logits / temperature, dim=-1)

            next_token = torch.multinomial(probs, num_samples=1)
            generated_tokens[:, i] = next_token.squeeze(dim=-1)

            next_token = torch.cat([next_token.unsqueeze(dim=1), next_token.unsqueeze(dim=1)], dim=1).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token)
            
            inputs_embeds = img_embeds.unsqueeze(dim=1)

    # print("generated_tokens:", generated_tokens)
    dec = model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec
    # print(visual_img.shape)
    picture = PIL.Image.fromarray(visual_img[0])

    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    ref_path = "/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/img_0.jpg"            # ← 换成你的本地图片路径
    ref_img  = PIL.Image.open(ref_path).convert("RGB").resize((img_size, img_size))
    def safe_ssim(imgA: np.ndarray, imgB: np.ndarray, max_pix=255):
    # 若尺寸过小则调 win_size；否则用默认
        h, w = imgA.shape[:2]
        if min(h, w) < 7:
            return structural_similarity(imgA, imgB,
                                        data_range=max_pix,
                                        channel_axis=-1,
                                        win_size=min(5, min(h, w)//2*2+1))
        else:
            return structural_similarity(imgA, imgB,
                                        data_range=max_pix,
                                        channel_axis=-1)

    # 转为 NumPy，float32 方便做差；visual_img 已是 uint8
    ref_np  = np.asarray(ref_img, dtype=np.float32)
    gen_np  = visual_img[0].astype(np.float32)

    # ─── 2. 计算 PSNR & SSIM ─────────────────────────────────────────────
    psnr_val = peak_signal_noise_ratio(ref_np, gen_np, data_range=255)
    ssim_val = safe_ssim(ref_np, gen_np)

    print(f"PSNR = {psnr_val:.2f} dB  |  SSIM = {ssim_val:.4f}")

    img  = clip_preprocess(picture).unsqueeze(0).cuda()
    # text = "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."

    text = clip.tokenize(['A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.']).cuda()
    with torch.no_grad():
        vi = clip_model.encode_image(img)
        vt = clip_model.encode_text(text)
        clip_val = torch.cosine_similarity(vi, vt).item()
    print(f'CLIP sim = {clip_val:.3f}')
    alpha, beta, gamma = 0.40, 0.35, 0.25
    
    return alpha * clip_val + beta * ssim_val + gamma * psnr_val


def get_layer_drop_config(removed_state) -> List[str]:
    # 获取removed_state中attn的长度
    num_blocks = len(removed_state["attn"])
    # 初始化drop_config为长度为num_blocks的列表，每个元素为"none"
    drop_config = ["none"] * num_blocks
    # 遍历num_blocks
    for i in range(num_blocks):
        # 如果removed_state中attn和mlp的对应元素都为True，则drop_config对应元素为"attn+mlp"
        if removed_state["attn"][i] and removed_state["mlp"][i]:
            drop_config[i] = "attn+mlp"
        # 如果removed_state中attn的对应元素为True，则drop_config对应元素为"attn"
        elif removed_state["attn"][i]:
            drop_config[i] = "attn"
        # 如果removed_state中mlp的对应元素为True，则drop_config对应元素为"mlp"
        elif removed_state["mlp"][i]:
            drop_config[i] = "mlp"
    # 返回drop_config
    return drop_config

def get_legal_mask(legal_to_drop_path, num_blocks):
    # 如果legal_to_drop_path为None，则返回一个字典，其中attn和mlp的值都为True
    if legal_to_drop_path is None:
        legal_to_drop = {"attn": [True] * num_blocks, "mlp": [True] * num_blocks}
        return legal_to_drop

    # 打开legal_to_drop_path文件，读取所有行
    with open(legal_to_drop_path, "r") as file:
        lines = file.readlines()
    # 去除每行末尾的换行符
    lines = [line.strip() for line in lines]

    # 断言lines的长度和num_blocks相等，如果不相等，则抛出异常
    assert (
        len(lines) == num_blocks
    ), "Number of blocks in model and legal_to_drop file do not match (If two_consecutive is set, number of blocks should be half of the model)"

    # 初始化legal_to_drop字典，其中attn和mlp的值都为False
    legal_to_drop = {"attn": [False] * len(lines), "mlp": [False] * len(lines)}
    # 遍历lines，根据每行的值设置legal_to_drop字典中对应位置的值
    for i in range(len(lines)):
        if lines[i] == "attn+mlp":
            legal_to_drop["attn"][i] = True
            legal_to_drop["mlp"][i] = True
        elif lines[i] == "attn":
            legal_to_drop["attn"][i] = True
        elif lines[i] == "mlp":
            legal_to_drop["mlp"][i] = True
    # 返回legal_to_drop字典
    return legal_to_drop

# 定义一个函数，用于判断一个状态是否有效
def is_valid_state(removed_state, legal_to_drop):
    # 遍历所有子块类型
    for subblock_type in ["attn", "mlp"]:
        # 遍历每个子块类型中的所有子块
        for i in range(len(legal_to_drop[subblock_type])):
            # 如果当前子块不能被删除，但是被删除了，则返回False
            if not legal_to_drop[subblock_type][i] and removed_state[subblock_type][i]:
                return False
    # 如果所有子块都符合条件，则返回True
    return True


def load_states(model, layers, removed_state, drop_two_consecutive):
    # 深度拷贝removed_state
    removed_state = copy.deepcopy(removed_state)
    # 如果drop_two_consecutive为True，则将removed_state中的attn和mlp列表中的每个元素都复制一遍
    if drop_two_consecutive:  # decompress: duplicate every entry
        removed_state["attn"] = [removed_state["attn"][i // 2] for i in range(2 * len(removed_state["attn"]))]
        removed_state["mlp"] = [removed_state["mlp"][i // 2] for i in range(2 * len(removed_state["mlp"]))]

    # 遍历removed_state中的attn和mlp列表
    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            # 根据subblock_type获取对应的subblock
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model.language_model.model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model.language_model.model))
            # 如果removed_state[subblock_type][j]为True，则将subblock设置为dummy_forward
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            # 否则，将subblock恢复为正常的forward
            else:
                restore_forward(subblock)

def compute_fitness(model, model_full, clip_model, clip_preprocess, prompt, data, fitness_fn, invert_fitness, target_logits: Optional[torch.Tensor] = None) -> float:
    # 定义一个变量sign，默认为1
    sign = 1
    # 如果invert_fitness为True，则将sign赋值为-1
    if invert_fitness:
        sign = -1

    # 如果fitness_fn为"ppl"，则调用compute_perplexity函数计算perplexity
    if fitness_fn == "ppl":
        return sign * compute_clip(model, model_full, clip_model, clip_preprocess, prompt)
    # 否则，调用compute_kl_div函数计算kl_div
    


# def selection(
#     model,
#     layers,
#     candidates,
#     num_survive: int,
#     calibration_data,
#     num_tokens: int,
#     drop_two_consecutive: bool,
#     invert_fitness: bool,
#     fitness_fn: str = "ppl",
#     target_logits: Optional[List[torch.Tensor]] = None,
# ):
#     # 初始化 calibration_minibatch、minibatch_ids、target_logits_minibatch 和 tokens_used
#     calibration_minibatch = []
#     minibatch_ids = []
#     target_logits_minibatch = []
#     tokens_used = 0
#     while tokens_used < num_tokens:  # generate minibatch with exactly num_tokens tokens
#         minibatch_id = random.randint(0, len(calibration_data) - 1)
#         if minibatch_id in minibatch_ids:  # avoid duplicates
#             continue
#         minibatch_ids.append(minibatch_id)
#         if tokens_used + calibration_data[minibatch_id].shape[1] > num_tokens:
#             calibration_minibatch.append(calibration_data[minibatch_id][:, : num_tokens - tokens_used])
#             if fitness_fn == "kl":
#                 target_logits_minibatch.append(target_logits[minibatch_id][:, : num_tokens - tokens_used])
#             tokens_used = num_tokens
#         else:
#             calibration_minibatch.append(calibration_data[minibatch_id])
#             if fitness_fn == "kl":
#                 target_logits_minibatch.append(target_logits[minibatch_id])
#             tokens_used += calibration_data[minibatch_id].shape[1]

#     if len(target_logits_minibatch) == 0:
#         target_logits_minibatch = None
#     fitnesses = []
#     for candidate in candidates:
#         load_states(model.language_model.model, layers, candidate, drop_two_consecutive)
#         fitness = compute_fitness(model, calibration_minibatch, fitness_fn, invert_fitness, target_logits_minibatch)
#         fitnesses.append(fitness)
#     # Keep only best
#     best_ids = np.argsort(fitnesses)[:num_survive]
#     return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def selection(
    model,
    model_full,
    clip_model,
    clip_preprocess,
    layers,
    candidates,
    prompt,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    drop_two_consecutive: bool,
    invert_fitness: bool,
    fitness_fn: str = "ppl",
):
    ## 选定测试数据
    test_data = calibration_data[:,:num_tokens]

    fitnesses = []
    for candidate in candidates:
        load_states(model, layers, candidate, drop_two_consecutive)
        fitness = compute_fitness(model, model_full, clip_model, clip_preprocess, prompt, test_data, fitness_fn, invert_fitness)
        fitnesses.append(fitness)

    # Keep only best
    best_ids = np.argsort(fitnesses)[:num_survive]
    return [candidates[i] for i in best_ids], [fitnesses[i] for i in best_ids]


def parse_args():
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="/mnt/temp/hshi/SD3/Janus-Pro-7B",
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
        default=None,
        help="The name or dataset or path used for calibration.",
    )
    parser.add_argument("--calibration_tokens", type=int, default=None, help="Number of tokens for calibration.")
    parser.add_argument(
        "--calibration_sequence_length", type=int, default=None, help="Length of calibration sequences."
    )
    parser.add_argument(
        "--eval_datasets",
        nargs="+",
        type=str,
        default=None,
        help="Datasets used for evaluation",
    )
    parser.add_argument("--no_eval", action="store_true", help="Whether to skip evaluation")
    parser.add_argument("--eval_every", default=1, type=int, help="Eval every # generations.")
    parser.add_argument("--eval_tokens", default=524288, type=int, help="Number of tokens for evaluation.")
    parser.add_argument("--eval_sequence_length", default=None, type=int, help="Length of evaluation sequences.")
    # Sparsification params
    parser.add_argument("--sparsity", type=float, default=0.15, help="Fraction of layers to drop.")
    # Logging params
    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search paramsss
    parser.add_argument("--fitness_fn", choices=["ppl", "kl"], default="ppl", help="Fitness function.")
    parser.add_argument("--generations", default=150, help="Number of generations in evolutionary search")
    parser.add_argument("--offspring", type=int, default=64, help="Number of offspring generated in each generation")
    parser.add_argument("--population_size", type=int, default=4, help="Population size in evolutionary search")
    parser.add_argument(
        "--initially_generated",
        type=int,
        default=64,
        help="Number of search points generated in the beginning; fittest are selected for the initial population",
    )
    parser.add_argument(
        "--initial_tokens",
        type=int,
        default=64,
        help="Number of calibration tokens used for the initial generation",
    )
    parser.add_argument(
        "--survivors_per_selection",
        type=int,
        nargs="+",
        default=[2, 4],
        help="Number of survivors after each stage of selection",
    )
    parser.add_argument(
        "--tokens_per_selection",
        type=int,
        nargs="+",
        default=[2048, 32768],
        help="Number of calibration tokens at each stage of selection",
    )
    # Evolutionary Search ablation params
    parser.add_argument(
        "--invert_fitness", default=True, help="Whether to invert the fitness function (search for worst)"
    )
    parser.add_argument("--max_mutations", type=int, default=3, help="Maximum number of mutations in offspring")
    parser.add_argument(
        "--legal_to_drop_path",
        type=str,
        default=None,
        help="Path to legal_to_drop file. A block can only be dropped if it is dropped in legal_to_drop configuration.",
    )
    parser.add_argument("--drop_entire_block", action="store_true", help="Whether to drop entire block (attn+mlp).")
    parser.add_argument(
        "--drop_two_consecutive",
        action="store_true",
        help="Only drop pairs of consecutive blocks (first and second, third and fourth,...). Can only be set when entire blocks are dropped.",
    )
    # Misc params
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "float32", "bfloat16"],
        help="dtype to load the model.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="Attention implementation: eager, sdpa, or flash_attention_2",
    )
    parser.add_argument("--use_fast_tokenizer", action="store_true", help="Whether to use fast tokenizer.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    # Save params
    parser.add_argument("--save_dir", type=str, default="/mnt/temp/hshi/EvoPress/EvoPress", help="Where to save sparse model.")
    parser.add_argument("--drop_config_dir", type=str, help="Where to save layer drop config.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    assert len(args.survivors_per_selection) == len(
        args.tokens_per_selection
    ), "Lists for selection survivors and tokens must have same length"
    assert args.survivors_per_selection[-1] == args.population_size, "Last stage should have population_size survivor"
    if args.drop_two_consecutive:
        assert args.drop_entire_block, "Can't drop two consecutive without dropping entire block"
        assert args.legal_to_drop_path == None, "Not implemented"
    # Get device and dtype

    print(args.generations)
    device = f"cuda"

    import torch 
    fix_seed(args.seed)
    # random.seed(args.seed)


    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # 多GPU场景

    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_name_or_path)
    tokenizer = vl_chat_processor.tokenizer

    model_full = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()
    model_full = model_full.to(torch.bfloat16).cuda().eval()

    import torch, clip, PIL.Image as Image
    clip_model, clip_preprocess = clip.load('ViT-B/32')


    conversation = [
        {
            "role": "User",
            "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "Assistant", "content": ""},
    ]

    parallel_size = 1
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag

    input_ids = vl_chat_processor.tokenizer.encode(prompt)
    input_ids = torch.LongTensor(input_ids)

    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.int).cuda()
    # print("tokens shape:", tokens.shape)
    for i in range(parallel_size*2):
        tokens[i, :] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    prompt = model.language_model.get_input_embeddings()(tokens)



    layers = get_layers(model.language_model.model)
    
    blocks_to_remove = int(args.sparsity * len(layers))
    print(f"Removing {blocks_to_remove} blocks")
    total_blocks = len(layers)

    calibration_data = torch.load(f"/mnt/temp/hshi/EvoPress/EvoPress/generated_tokens.pt")

    eval_datasets = torch.load(f"/mnt/temp/hshi/EvoPress/EvoPress/generated_tokens.pt")

    if args.drop_two_consecutive:
        assert total_blocks % 2 == 0 and blocks_to_remove % 2 == 0, "Number of total and removed blocks must be even"
        total_blocks = total_blocks // 2  # view two consecutive blocks as one block
        blocks_to_remove = blocks_to_remove // 2

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model.language_model.model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model.language_model.model)))

    legal_mask = get_legal_mask(
        args.legal_to_drop_path, total_blocks
    )  # mask of blocks that can be dropped (all blocks by default)

    initial_population_candidates = (
        []
    )  # store initially generated search points (only take fittest for first population)

    while len(initial_population_candidates) < args.initially_generated:
        removed_state = {"attn": [False] * total_blocks, "mlp": [False] * total_blocks}

        attn_legal_ind = [i for i in range(total_blocks) if legal_mask["attn"][i]]
        attn_remove_ind = random.sample(attn_legal_ind, blocks_to_remove)
        for ind in attn_remove_ind:
            removed_state["attn"][ind] = True

        mlp_legal_ind = [i for i in range(total_blocks) if legal_mask["mlp"][i]]
        mlp_remove_ind = random.sample(mlp_legal_ind, blocks_to_remove)
        for ind in mlp_remove_ind:
            removed_state["mlp"][ind] = True

        if args.drop_entire_block:
            removed_state["mlp"] = copy.deepcopy(removed_state["attn"])

        if removed_state in initial_population_candidates:  # avoid duplicates
            continue
        if not is_valid_state(removed_state, legal_mask):
            continue

        initial_population_candidates.append(removed_state)

    population, train_fitnesses = selection(
        model=model,
        model_full=model_full,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        layers=layers,
        candidates=initial_population_candidates,
        prompt=prompt,
        num_survive=args.population_size,
        calibration_data=calibration_data,
        invert_fitness=args.invert_fitness,
        drop_two_consecutive=args.drop_two_consecutive,
        num_tokens=args.initial_tokens,
        fitness_fn=args.fitness_fn,
    )



    for gen_id in range(args.generations):
        print(f"Generation {gen_id + 1}/{args.generations}")
        print(f"Train fitness {train_fitnesses[0]}")

        for parent in population:
            print(f"Parent: attn: {[int(ele) for ele in parent['attn']]} mlp: {[int(ele) for ele in parent['mlp']]}")

        load_states(model, layers, population[0], args.drop_two_consecutive)

        # Evaluate current search point
        if gen_id % args.eval_every == 0 and not args.no_eval:
            # ppl_eval = compute_clip(model, prompt, eval_datasets)
            ppl_eval = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)
            # full_train_ppl = compute_clip(model, prompt, calibration_data)
            full_train_ppl = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)

        offspring_list = []

        # Generate offspring by Mutation
        while len(offspring_list) < args.offspring:
            offspring = copy.deepcopy(random.choice(population))

            # Mutation
            num_flips = min(
                random.randint(1, args.max_mutations), random.randint(1, args.max_mutations)
            )  # bias towards lower values
            for _ in range(num_flips):
                remove_type = random.randint(0, 1)  # 0 remove attention, 1 remove mlp
                if remove_type == 0:
                    subblock_type = "attn"
                else:
                    subblock_type = "mlp"

                remove_ind = random.randint(0, total_blocks - 1)
                while offspring[subblock_type][remove_ind]:
                    remove_ind = random.randint(0, total_blocks - 1)

                add_ind = random.randint(0, total_blocks - 1)
                while not offspring[subblock_type][add_ind]:
                    add_ind = random.randint(0, total_blocks - 1)

                offspring[subblock_type][remove_ind] = True
                offspring[subblock_type][add_ind] = False

            if args.drop_entire_block:
                offspring["mlp"] = copy.deepcopy(offspring["attn"])

            if offspring in offspring_list or offspring in population:  # avoid duplicates
                continue

            if not is_valid_state(offspring, legal_mask):
                continue

            offspring_list.append(offspring)

        # Selection in multiple steps
        for num_survive, num_tokens in zip(args.survivors_per_selection, args.tokens_per_selection):
            if num_survive == args.survivors_per_selection[-1]:
                for i in range(
                    len(population)
                ):  # Elitist EA: Add search points in current generation to final selection step
                    if population[i] not in offspring_list:
                        offspring_list.append(population[i])

            offspring_list, train_fitnesses = selection(
                model=model,
                model_full=model_full,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                layers=layers,
                candidates=offspring_list,
                prompt=prompt,
                num_survive=num_survive,
                calibration_data=calibration_data,
                drop_two_consecutive=args.drop_two_consecutive,
                invert_fitness=args.invert_fitness,
                num_tokens=num_tokens,
                fitness_fn=args.fitness_fn,
            )

        population = offspring_list

        layer_drop_config = get_layer_drop_config(population[0])
        if args.drop_config_dir:
            os.makedirs(args.drop_config_dir, exist_ok=True)
            with open(os.path.join(args.drop_config_dir, "layer_drop_config.txt"), "w") as f:
                for line in layer_drop_config:
                    f.write(line + "\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save model
        torch.save(model.language_model.model, os.path.join(args.save_dir, "final_model.pth"))
        save_dir1 = os.path.join(args.save_dir, "modified_janus")
        model.save_pretrained(save_dir1)
        print("模型已保存至:", save_dir1)

        # Save layer drop config
        with open(os.path.join(args.save_dir, "layer_drop_config.txt"), "w") as f:
            for line in layer_drop_config:
                f.write(line + "\n")

    print("Final configuration:")
    for line in layer_drop_config:
        print(line)

    ppl_eval = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)
    full_train_ppl = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)

if __name__ == "__main__":
    main()








    