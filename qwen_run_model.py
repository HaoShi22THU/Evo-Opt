from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

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

    removed_state = copy.deepcopy(removed_state)

    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            # 根据subblock_type获取对应的subblock
            # print("subblock_type:", subblock_type)
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model))
            # 如果removed_state[subblock_type][j]为True，则将subblock设置为dummy_forward
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            # 否则，将subblock恢复为正常的forward
            else:
                restore_forward(subblock)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("/mnt/temp/hshi/Qwen/Qwen2.5-VL-7B-Instruct")

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

for layer in layers:
    dummy_initialize(getattr(layer, get_attn_layer_name(model)))
    dummy_initialize(getattr(layer, get_mlp_layer_name(model)))

drop_config = load_drop_config()

load_states(model, layers, drop_config)

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
for _ in range(10):
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
print(output_text)
