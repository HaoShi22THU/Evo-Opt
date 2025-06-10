from transformers import modeling_utils
if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
    # 只需填一个非空可迭代对象即可；这里给出官方目前支持的 4 种并行风格
    modeling_utils.ALL_PARALLEL_STYLES = {"tp", "none", "colwise", "rowwise"}

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import types
from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as qwen_vl

def _fixed_get_rope_index(self, input_ids, attention_mask, *args, **kwargs):
    """
    A drop‑in replacement for Qwen2_5_VLModel.get_rope_index that avoids the
    shape‑mismatch error when attention_mask contains vision tokens.
    """
    # When no mask is given, fall back to a simple arange.
    if attention_mask is None:
        seq_len = input_ids.size(1)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0).expand(input_ids.size(0), -1)
        return position_ids, None

    # Build position_ids sample‑by‑sample.
    pos_list = []
    for i in range(input_ids.size(0)):       # iterate over batch
        text_mask = attention_mask[i] == 1   # keep only text tokens
        pos = torch.arange(
            text_mask.sum(),
            dtype=torch.long,
            device=input_ids.device,
        )
        pos_list.append(pos)

    # Pad to the longest sequence in the batch.
    position_ids = torch.nn.utils.rnn.pad_sequence(
        pos_list, batch_first=True, padding_value=0
    )
    return position_ids, None  # rope_deltas is unchanged

# Monkey‑patch the model class
qwen_vl.Qwen2_5_VLModel.get_rope_index = types.MethodType(
    _fixed_get_rope_index, qwen_vl.Qwen2_5_VLModel
)

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import numpy as np

import PIL.Image
import copy

from src.model_utils import (
    get_layers,
    get_attn_layer_name,
    get_mlp_layer_name,
    make_dummy_forward,
    dummy_initialize,
    restore_forward,
)

def load_drop_config():
    with open("/mnt/temp/hshi/EvoPress/EvoPress/layer_skip_config.txt", "r") as f:
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

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


layers = get_layers(model)
print(f"Number of layers: {len(layers)}")

for layer in layers:
    dummy_initialize(getattr(layer, get_attn_layer_name(model.model.language_model)))
    dummy_initialize(getattr(layer, get_mlp_layer_name(model.model.language_model)))

# 读取配置文件
drop_config = load_drop_config()

load_states(model, layers, drop_config)


# Preparation for inference
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

# Inference: Generation of the output
for _ in range(10):
    generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text)
