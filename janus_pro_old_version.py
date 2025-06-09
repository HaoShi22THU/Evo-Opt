import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

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
    with open("/mnt/temp/hshi/EvoPress/EvoPress/layer_drop_config_007.txt", "r") as f:
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
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_name_or_path)
    tokenizer = vl_chat_processor.tokenizer

    model = model.to(torch.bfloat16).cuda().eval()

    conversation = [
        {
            "role": "User",
            "content": "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue.",
        },
        {"role": "Assistant", "content": ""},
    ]

    parallel_size = 2
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

    inputs_embeds = model.language_model.get_input_embeddings()(tokens)

    layers = get_layers(model.language_model.model)

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model.language_model.model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model.language_model.model)))

    # 读取配置文件
    drop_config = load_drop_config()
    # print("drop_config:", drop_config)

    # 加载状态
    load_states(model, layers, drop_config)
    img_size = 384
    patch_size = 16
    temperature = 1
    cfg_weight = 5
    generated_tokens = torch.zeros((parallel_size, 576), dtype=torch.long).cuda()

    for i in range(576):
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

    print(generated_tokens)

    dec = model.gen_vision_model.decode_code(generated_tokens.to(dtype=torch.int), shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size])
    dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)

    dec = np.clip((dec + 1) / 2 * 255, 0, 255)

    visual_img = np.zeros((parallel_size, img_size, img_size, 3), dtype=np.uint8)
    visual_img[:, :, :] = dec

    os.makedirs('generated_samples', exist_ok=True)
    for i in range(parallel_size):
        save_path = os.path.join('generated_samples', "img_{}_revise_007_1.jpg".format(i))
        PIL.Image.fromarray(visual_img[i]).save(save_path)

if __name__ == "__main__":
    main()

