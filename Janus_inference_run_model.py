import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from janus.utils.io import load_pil_images
from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
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
# torch.backends.cuda.cudnn.benchmark = True  # 动态选择高效算法
# torch.backends.cuda.memory_snapshot = False  # 禁用内存快照
@torch.no_grad()
def load_states(model, layers, removed_state):

    removed_state = copy.deepcopy(removed_state)

    for subblock_type in ["attn", "mlp"]:
        for j in range(len(removed_state[subblock_type])):
            # 根据subblock_type获取对应的subblock
            # print("subblock_type:", subblock_type)
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

def load_drop_config():
    with open("/mnt/temp/hshi/EvoPress/EvoPress/layer_drop_config_inference_30.txt", "r") as f:
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
def main():
    seed=0

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多GPU场景

    model_name_or_path = "/mnt/temp/hshi/SD3/Janus-Pro-7B"
    model_path = "/mnt/temp/hshi/SD3/Janus-Pro-7B"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True
    )
    model = model.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "User",
            "content": "<image_placeholder>\nUnderstand this image.",
            "images": ["/mnt/temp/hshi/EvoPress/EvoPress/generated_samples/img_0.jpg"],
        },
        {"role": "Assistant", "content": ""},
    ]

    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation, images=pil_images, force_batchify=True
    ).to(model.device)

    # # run image encoder to get the image embeddings
    inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

    layers = get_layers(model.language_model.model)

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model.language_model.model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model.language_model.model)))
    drop_config = load_drop_config()
    load_states(model, layers, drop_config)

    # # run the model to get the response
    outputs = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True,
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    print(f"{prepare_inputs['sft_format'][0]}", answer)




if __name__ == "__main__":
    main()



