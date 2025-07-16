from transformers import modeling_utils
if getattr(modeling_utils, "ALL_PARALLEL_STYLES", None) is None:
    # 只需填一个非空可迭代对象即可；这里给出官方目前支持的 4 种并行风格
    modeling_utils.ALL_PARALLEL_STYLES = {"tp", "none", "colwise", "rowwise"}

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info



import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from src.model_utils import (
    get_layers_vit,
    get_attn_layer_name_vit,
    get_mlp_layer_name_vit,
)

def load_drop_config():
    with open("/mnt/temp/hshi/EvoPress/EvoPress/qwen_layer_skip_config_inference_40_20_sep.txt", "r") as f:
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
def load_states(model, layers, blocks, removed_state):
    # 深度拷贝removed_state
    removed_state = copy.deepcopy(removed_state)

    # 遍历removed_state中的attn和mlp列表
    for subblock_type in ["attn", "mlp"]:
        for j in range(len(layers)):
            # 根据subblock_type获取对应的subblock
            if subblock_type == "attn":
                subblock = getattr(layers[j], get_attn_layer_name(model.model))
            else:
                subblock = getattr(layers[j], get_mlp_layer_name(model.model))
            # 如果removed_state[subblock_type][j]为True，则将subblock设置为dummy_forward
            if removed_state[subblock_type][j]:
                make_dummy_forward(subblock, subblock_type)
            # 否则，将subblock恢复为正常的forward
            else:
                restore_forward(subblock)
    
    for subblock_type in ["attn", "mlp"]:
        for j in range(len(blocks)):
            if subblock_type == "attn":
                subblock = getattr(blocks[j], get_attn_layer_name_vit(model.model))
            else:
                subblock = getattr(blocks[j], get_mlp_layer_name_vit(model.model))
            # 如果removed_state[subblock_type][j]为True，则将subblock设置为dummy_forward
            if removed_state[subblock_type][j+len(layers)]:
                make_dummy_forward(subblock, subblock_type)
            # 否则，将subblock恢复为正常的forward
            else:
                restore_forward(subblock)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "/mnt/temp/hshi/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto", trust_remote_code=True
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processer
processor = AutoProcessor.from_pretrained("/mnt/temp/hshi/Qwen2.5-VL-7B-Instruct")

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
                # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                "image": "/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/demo1.png",
            },
            {"type": "text", "text": "Describe this image."},
        ],
    }
]


layers = get_layers(model)
blocks = get_layers_vit(model)

print(f"Number of layers: {len(layers)}")
print(f"Number of blocks: {len(blocks)}")

for layer in layers:
    dummy_initialize(getattr(layer, get_attn_layer_name(model.model.language_model)))
    dummy_initialize(getattr(layer, get_mlp_layer_name(model.model.language_model)))
    
for block in blocks:
    dummy_initialize(getattr(block, get_attn_layer_name_vit(model)))
    dummy_initialize(getattr(block, get_mlp_layer_name_vit(model)))


# 读取配置文件
drop_config = load_drop_config()

load_states(model, layers, blocks, drop_config)

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

generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache=True)


with torch.no_grad():
    prompt_len = inputs.input_ids.shape[1]

    seq_ids   = generated_ids                          # prompt + 新文本
    seq_attn  = torch.ones_like(seq_ids)               # 全 1 mask
    labels    = seq_ids.clone()
    labels[:, :prompt_len] = -100                      # 只评估生成段

    # 1) 复制一份原 inputs
    lm_inputs = {k: v for k, v in inputs.items()}
    # 2) 替换文本相关字段
    lm_inputs.update({
        "input_ids":      seq_ids,
        "attention_mask": seq_attn,
        "labels":         labels,
    })

    out  = model(**lm_inputs, use_cache=False, return_dict=True)
    ppl  = torch.exp(out.loss).item()

print(f"PPL on generated text: {ppl:.3f}")
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
