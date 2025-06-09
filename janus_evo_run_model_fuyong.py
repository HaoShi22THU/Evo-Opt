#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用 Janus-Pro-7B 做图像生成，并在推理阶段按 layer_drop_base.txt
配置跳过指定层的 attn / mlp，改为 **复用上一层已缓存的输出**，
而不是简单返回占位张量。
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"         # ← 视情况修改 GPU

import copy
import torch
import torch.nn.functional as F
import numpy as np
import PIL.Image

from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor

# ------------------------------------------------------------
# 0. 依赖的工具函数（保持使用你项目里的实现）
# ------------------------------------------------------------
from src.model_utils import (
    get_layers,
    get_attn_layer_name,
    get_mlp_layer_name,
)

# ------------------------------------------------------------
# 1. 复用-缓存相关逻辑
# ------------------------------------------------------------
_LAST_OUTPUT = {"attn": None, "mlp": None}       # 最近一次“有效”输出


def _attach_cache_writer(subblock, subblock_type: str):
    """给正常 sub-block 打补丁：执行完 forward 把输出写入缓存."""
    if hasattr(subblock, "__orig_forward"):
        return                                   # 已打过补丁

    subblock.__orig_forward = subblock.forward

    def forward_with_cache(self, *args, **kwargs):
        out = self.__orig_forward(*args, **kwargs)
        _LAST_OUTPUT[subblock_type] = out
        return out

    subblock.forward = forward_with_cache.__get__(subblock, subblock.__class__)


def _replace_with_reuse(subblock, subblock_type: str):
    """
    把 forward 改成“复用上一层输出”：
      • 缓存有值 → 直接返回  
      • 第一次用到但缓存仍为空 → 真算一次并写缓存
    """
    if not hasattr(subblock, "__orig_forward"):
        subblock.__orig_forward = subblock.forward

    def reuse_forward(self, *args, **kwargs):
        if _LAST_OUTPUT[subblock_type] is None:          # 需先填缓存
            out = self.__orig_forward(*args, **kwargs)
            _LAST_OUTPUT[subblock_type] = out
            return out
        return _LAST_OUTPUT[subblock_type]

    subblock.forward = reuse_forward.__get__(subblock, subblock.__class__)


def load_states(model, layers, removed_state):
    """
    根据 removed_state 给每层的 attn / mlp 打补丁：
      • 不跳过：attach_cache_writer（正常计算 + 写缓存）  
      • 跳过  ：reuse 前层缓存
    """
    removed_state = copy.deepcopy(removed_state)

    for subblock_type in ["attn", "mlp"]:
        for j in range(len(layers)):
            if subblock_type == "attn":
                subblock = getattr(layers[j],
                                   get_attn_layer_name(model.language_model.model))
            else:
                subblock = getattr(layers[j],
                                   get_mlp_layer_name(model.language_model.model))

            # 所有 sub-block 都要先有写缓存能力
            _attach_cache_writer(subblock, subblock_type)

            # 若该层被“drop”，则改为复用缓存
            if removed_state[subblock_type][j]:
                _replace_with_reuse(subblock, subblock_type)


# ------------------------------------------------------------
# 2. layer_drop_base.txt 读取工具
# ------------------------------------------------------------
def load_drop_config(txt_path: str):
    """
    文件格式：每行 one of {attn, mlp, attn+mlp, none}
    行号 = layer index
    """
    with open(txt_path, "r") as f:
        lines = f.readlines()

    removed_state = {"attn": [False] * len(lines),
                     "mlp":  [False] * len(lines)}

    for i, line in enumerate(lines):
        tag = line.strip()
        if tag == "attn":
            removed_state["attn"][i] = True
        elif tag == "mlp":
            removed_state["mlp"][i] = True
        elif tag == "attn+mlp":
            removed_state["attn"][i] = True
            removed_state["mlp"][i] = True
        elif tag == "none":
            pass
        else:
            raise ValueError(f"Invalid tag in layer_drop_base.txt: {tag}")
    return removed_state


# ------------------------------------------------------------
# 3. 主流程
# ------------------------------------------------------------
@torch.no_grad()
def main():
    # ---------------- 环境与模型 ----------------
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model_path = "/mnt/temp/hshi/SD3/Janus-Pro-7B"       # ← 按需修改
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(torch.bfloat16).cuda().eval()

    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    # --------------- 文本 Prompt ----------------
    conversation = [
        {
            "role": "User",
            "content": "A close-up high-contrast photo of Sydney Opera House "
                       "sitting next to Eiffel tower, under a blue night sky of "
                       "roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "Assistant", "content": ""},
    ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )
    prompt = sft_format + vl_chat_processor.image_start_tag
    input_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)

    # --------------- 生成参数 --------------------
    parallel_size  = 10
    img_size       = 384
    patch_size     = 16
    temperature    = 1.0
    cfg_weight     = 5.0
    max_tokens     = 576

    # --------------- 准备 tokens & embeds --------
    tokens = torch.zeros((parallel_size * 2, len(input_ids)), dtype=torch.long).cuda()
    for i in range(parallel_size * 2):
        tokens[i, :] = input_ids
        if i % 2 == 1:                               # 偶数组 uncond → pad 文本
            tokens[i, 1:-1] = vl_chat_processor.pad_id

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)

    # --------------- Layer patch -----------------
    layers = get_layers(model.language_model.model)
    drop_cfg = load_drop_config("/mnt/temp/hshi/EvoPress/EvoPress/layer_drop_config_007.txt")
    load_states(model, layers, drop_cfg)            # 打补丁

    # --------------- 逐 token 采样 ---------------
    generated = torch.zeros((parallel_size, max_tokens), dtype=torch.long).cuda()
    past_key_values = None

    for i in range(max_tokens):
        outputs = model.language_model.model(
            inputs_embeds=inputs_embeds,
            use_cache=True,
            past_key_values=past_key_values if i != 0 else None,
        )
        past_key_values = outputs.past_key_values
        logits = model.gen_head(outputs.last_hidden_state[:, -1, :])

        # classifier-free guidance
        logit_cond  = logits[0::2, :]
        logit_uncond = logits[1::2, :]
        logits = logit_uncond + cfg_weight * (logit_cond - logit_uncond)

        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)         # (B, 1)
        generated[:, i] = next_token.squeeze(1)

        # prepare next img embed
        next_cat = torch.cat([next_token, next_token], dim=1).view(-1)  # cond+uncond
        inputs_embeds = model.prepare_gen_img_embeds(next_cat).unsqueeze(1)

    # --------------- 解码 & 保存 ------------------
    dec = model.gen_vision_model.decode_code(
        generated.to(torch.int),
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size],
    )
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    os.makedirs("generated_samples", exist_ok=True)
    for idx, img in enumerate(dec):
        PIL.Image.fromarray(img).save(f"generated_samples/img_{idx}_fy_007.jpg")
    print("✅ 生成完成，图像已保存在 generated_samples/")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
