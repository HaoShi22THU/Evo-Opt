# # 安装依赖（如已安装可跳过）
# # pip install torch transformers --upgrade

# import torch
# from transformers import CLIPProcessor, CLIPModel
# import clip
# # 1. 加载 CLIP 模型（ViT-B/32 骨干）和处理器
# model, processor = clip.load('ViT-B/32')

# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = model.to(device)

# def safe_tokenize(texts, max_length=76):
#         flat = []
#         for t in texts:
#             if isinstance(t, (list, tuple)):
#                 flat.extend(t)       # 把 list[str] 展平
#             else:
#                 flat.append(t)
#         tokens = clip.tokenize(flat, truncate=True)
#         return tokens.to(device)

    


# def clip_text_similarity(t1: str, t2: str) -> float:
#     with torch.no_grad():
#         # ① tokenize → (2, max_len) int tensor

#         text_inputs = safe_tokenize([t1, t2])
#         # tokens = clip.tokenize(text_inputs).to(device)

#         # ② 得到文本特征 (2, 512)
#         embeds = model.encode_text(text_inputs)

#         # ③ 归一化 & 余弦
#         embeds = embeds / embeds.norm(dim=-1, keepdim=True)
#         sim = (embeds[0] @ embeds[1]).item()       # 点积就是余弦
#     return sim

# # ========= 示例 =========
# if __name__ == "__main__":
#     t1 = " 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、s， 、 有 、s， 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有 有  有 有 有 "
#     t2 = "addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n addCriterion\n"
#     sim = clip_text_similarity(t1, t2)
#     print(f"相似度: {sim:.4f}")   # 值越接近 1 语义越相似

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

from transformers import modeling_utils

if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
    # 只需填一个非空可迭代对象即可；这里给出官方目前支持的 4 种并行风格
    modeling_utils.ALL_PARALLEL_STYLES = {"tp", "none", "colwise", "rowwise"}

# 之后再导入 / 加载 Qwen2_5_VLForConditionalGeneration
from transformers import Qwen2_5_VLForConditionalGeneration

import types
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_vl

# def _fixed_get_rope_index(self, input_ids, attention_mask, *args, **kwargs):
#     """
#     A drop‑in replacement for Qwen2_5_VLModel.get_rope_index that avoids the
#     shape‑mismatch error when attention_mask contains vision tokens.
#     """
#     # When no mask is given, fall back to a simple arange.
#     if attention_mask is None:
#         seq_len = input_ids.size(1)
#         position_ids = torch.arange(
#             seq_len, dtype=torch.long, device=input_ids.device
#         ).unsqueeze(0).expand(input_ids.size(0), -1)
#         return position_ids, None

#     # Build position_ids sample‑by‑sample.
#     pos_list = []
#     for i in range(input_ids.size(0)):       # iterate over batch
#         text_mask = attention_mask[i] == 1   # keep only text tokens
#         pos = torch.arange(
#             text_mask.sum(),
#             dtype=torch.long,
#             device=input_ids.device,
#         )
#         pos_list.append(pos)

#     # Pad to the longest sequence in the batch.
#     position_ids = torch.nn.utils.rnn.pad_sequence(
#         pos_list, batch_first=True, padding_value=0
#     )
#     return position_ids, None  # rope_deltas is unchanged

# Monkey‑patch the model class
# qwen_vl.Qwen2_5_VLModel.get_rope_index = types.MethodType(
#     _fixed_get_rope_index, qwen_vl.Qwen2_5_VLModel
# )

from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import  AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

from src.data_utils import get_data
from src.common_utils import fix_seed
from janus.utils.io import load_pil_images
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

def compute_clip(model, model_full, processor, clip_model, clip_preprocess, prompt, base_answer):
    device = next(model.parameters()).device

    # model_path = "/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct"

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    # # print(model.model)
    # processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")
    # inputs.pop("attention_mask", None)

    generated_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        use_cache=True
    )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answer = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    

    def safe_tokenize(texts, max_length=76):
        flat = []
        for t in texts:
            if isinstance(t, (list, tuple)):
                flat.extend(t)       # 把 list[str] 展平
            else:
                flat.append(t)
        tokens = clip.tokenize(flat, truncate=True)
        return tokens.to(device)

    text_inputs = safe_tokenize([answer, base_answer])
    
    # 提取特征向量
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs)

    # 归一化
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # 计算相似度
    similarity = (text_features[0] @ text_features[1].T).item()

    print('similarity:', similarity)
   
    return similarity


def load_drop_config():
    with open("/mnt/temp/hshi/EvoPress/EvoPress/qwen_layer_skip_config_inference_40_L.txt", "r") as f:
        lines = f.readlines()
    # 读取每一行的内容
    removed_state = {"attn": [False] * len(lines), "mlp": [False] * len(lines)}
    print(lines)
    for i in range(len(lines)):
        # 去除行首尾空格
        line = lines[i]
        # 如果行不为空，则进行处理
        # print(line)
        if line == "attn\n":
            removed_state["attn"][i] = True
        elif line == "mlp\n":
            removed_state["mlp"][i] = True
        elif line == "attn+mlp\n":
            removed_state["attn"][i] = True
            removed_state["mlp"][i] = True
        elif line == "none\n":
            removed_state["attn"][i] = False
            removed_state["mlp"][i] = False
        else:
            print("error: invalid line in layer_drop_config.txt")

        # print("removed_state:", removed_state)

    return removed_state


@torch.no_grad()
def load_states(model, layers, removed_state):
    # 深度拷贝removed_state
    removed_state = copy.deepcopy(removed_state)
    # 如果drop_two_consecutive为True，则将removed_state中的attn和mlp列表中的每个元素都复制一遍
    # if drop_two_consecutive:  # decompress: duplicate every entry
    #     removed_state["attn"] = [removed_state["attn"][i // 2] for i in range(2 * len(removed_state["attn"]))]
    #     removed_state["mlp"] = [removed_state["mlp"][i // 2] for i in range(2 * len(removed_state["mlp"]))]

    # 遍历removed_state中的attn和mlp列表
    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            # 根据subblock_type获取对应的subblock
            # print("subblock_type:", subblock_type)
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model.model.language_model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model.model.language_model))
            # 如果removed_state[subblock_type][j]为True，则将subblock设置为dummy_forward
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            # 否则，将subblock恢复为正常的forward
            else:
                restore_forward(subblock)


def main():
    model_path = "/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    print(model.model)
    processor = AutoProcessor.from_pretrained(model_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True)

    # generated_ids_trimmed = [
    #     out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    # ]
    # base_answers = processor.batch_decode(
    #     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    # )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    base_answers = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    import torch, clip, PIL.Image as Image
    clip_model, clip_preprocess = clip.load('ViT-B/32')

    layers = get_layers(model)
    print(f"Number of layers: {len(layers)}")

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model.model.language_model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model.model.language_model)))

    # 读取配置文件
    drop_config = load_drop_config()

    load_states(model, layers, drop_config)
    clip_socre = compute_clip(model, model, processor, clip_model, clip_preprocess, text, base_answers)

    print(clip_socre)

if __name__ == "__main__":
    main()