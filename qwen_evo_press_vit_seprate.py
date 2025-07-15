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

from src.model_utils import (
    get_layers_vit,
    get_attn_layer_name_vit,
    get_mlp_layer_name_vit,
)

### Janus Pro
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor
import numpy as np
import os
import PIL.Image

@torch.no_grad()
def compute_clip(model, model_full, processor, clip_model, clip_preprocess, prompt, base_answer):
    device = next(model.parameters()).device

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
    # return -1

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

def load_states(model, layers, blocks,removed_state):
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

def compute_fitness(model, model_full, processor, clip_model, clip_preprocess, prompt, base_answers, data, fitness_fn, invert_fitness, target_logits: Optional[torch.Tensor] = None) -> float:
    # 定义一个变量sign，默认为1
    sign = 1
    # 如果invert_fitness为True，则将sign赋值为-1
    if invert_fitness:
        sign = -1

    # 如果fitness_fn为"ppl"，则调用compute_perplexity函数计算perplexity
    if fitness_fn == "ppl":
        return sign * compute_clip(model, model_full, processor, clip_model, clip_preprocess, prompt, base_answers)
    # 否则，调用compute_kl_div函数计算kl_div

def selection(
    model,
    model_full,
    processor,
    clip_model,
    clip_preprocess,
    layers,
    candidates,
    prompt,
    base_answer,
    num_survive: int,
    calibration_data,
    num_tokens: int,
    drop_two_consecutive: bool,
    invert_fitness: bool,
    fitness_fn: str = "ppl",
):
    ## 选定测试数据
    test_data = calibration_data[:,:num_tokens]

    blocks = get_layers_vit(model)

    fitnesses = []
    for candidate in candidates:
        load_states(model, layers, blocks, candidate)
        # pristine_prompt = {k: v.clone() if isinstance(v, torch.Tensor) else copy.deepcopy(v)
        #                for k, v in prompt.items()} 
        fitness = compute_fitness(model, model_full, processor, clip_model, clip_preprocess, prompt, base_answer, test_data, fitness_fn, invert_fitness)
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
    parser.add_argument("--sparsity", type=float, default=0.40, help="Fraction of layers to drop.")
    # Logging params

    parser.add_argument("--Vit_sparsity", type=float, default=0.40, help="Fraction of blocks to drop in ViT (32 layers).")
    parser.add_argument("--Language_sparsity", type=float, default=0.40, help="Fraction of layers to drop in Language model (28 layers).")

    parser.add_argument("--log_wandb", default=False, action="store_true", help="Whether to log to W&B")
    # Evolutionary Search paramsss
    parser.add_argument("--fitness_fn", choices=["ppl", "kl"], default="ppl", help="Fitness function.")
    parser.add_argument("--generations", default=50, help="Number of generations in evolutionary search")
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
    parser.add_argument("--drop_config_dir", type=str, default="/mnt/temp/hshi/EvoPress/EvoPress", help="Where to save layer drop config.")

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

    print(args.generations)
    device = f"cuda"

    import torch 
    fix_seed(args.seed)

    model_path = "/mnt/temp/hshi/Qwen2.5-VL-7B-Instruct"

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_path, torch_dtype="auto", device_map="auto", trust_remote_code=True)

    print(model.model)
    processor = AutoProcessor.from_pretrained(model_path)

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

    generated_ids = model.generate(**inputs, max_new_tokens=128)

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
    blocks = get_layers_vit(model)

    layers_to_remove = int(args.Language_sparsity * len(layers))
    print(f"Removing {layers_to_remove} layers")
    blocks_to_remove = int(args.Vit_sparsity * len(blocks))
    print(f"Removing {blocks_to_remove} blocks")
    
    # blocks_to_remove = int(args.sparsity * (len(layers) + len(blocks)))
    # print(f"Removing {blocks_to_remove} blocks")

    total_blocks = len(layers) + len(blocks)

    calibration_data = torch.load(f"/mnt/temp/hshi/EvoPress/EvoPress/generated_tokens.pt")

    eval_datasets = torch.load(f"/mnt/temp/hshi/EvoPress/EvoPress/generated_tokens.pt")

    for layer in layers:
        dummy_initialize(getattr(layer, get_attn_layer_name(model.model.language_model)))
        dummy_initialize(getattr(layer, get_mlp_layer_name(model.model.language_model)))

    for block in blocks:
        dummy_initialize(getattr(block, get_attn_layer_name_vit(model)))
        dummy_initialize(getattr(block, get_mlp_layer_name_vit(model)))

    legal_mask_vit = get_legal_mask(
        args.legal_to_drop_path, blocks_to_remove
    )  # mask of blocks that can be dropped (all blocks by default)

    legal_mask_language = get_legal_mask(
        args.legal_to_drop_path, layers_to_remove
    )  # mask of blocks that can be dropped (all blocks by default)

    initial_population_candidates = (
        []
    )  # store initially generated search points (only take fittest for first population)

    while len(initial_population_candidates) < args.initially_generated:
        removed_state_vit = {"attn": [False] * blocks_to_remove, "mlp": [False] * blocks_to_remove}
        removed_state_language = {"attn": [False] * layers_to_remove, "mlp": [False] * layers_to_remove}

        attn_legal_ind_vit = [i for i in range(blocks_to_remove) if legal_mask_vit["attn"][i]]
        attn_remove_ind_vit = random.sample(attn_legal_ind_vit, blocks_to_remove)
        for ind in attn_remove_ind_vit:
            removed_state_vit["attn"][ind] = True

        mlp_legal_ind = [i for i in range(blocks_to_remove) if legal_mask_vit["mlp"][i]]
        mlp_remove_ind = random.sample(mlp_legal_ind, blocks_to_remove)
        for ind in mlp_remove_ind:
            removed_state_vit["mlp"][ind] = True


        attn_legal_ind_language = [i for i in range(layers_to_remove) if legal_mask_language["attn"][i]]
        attn_remove_ind_language = random.sample(attn_legal_ind_language, layers_to_remove)
        for ind in attn_remove_ind_language:
            removed_state_language["attn"][ind] = True
        
        mlp_legal_ind_language = [i for i in range(layers_to_remove) if legal_mask_language["mlp"][i]]
        mlp_remove_ind_language = random.sample(mlp_legal_ind_language, layers_to_remove)
        for ind in mlp_remove_ind_language:
            removed_state_language["mlp"][ind] = True
    
        ## 合并两个字典
        removed_state = {
            "attn": removed_state_language["attn"] + removed_state_vit["attn"],
            "mlp": removed_state_language["mlp"] + removed_state_vit["mlp"],
        }

        if removed_state in initial_population_candidates:  # avoid duplicates
            continue

        if removed_state["attn"][0] != False or removed_state["attn"][28] != False:
            # print("Error: first block cannot be removed")
            continue

        initial_population_candidates.append(removed_state)

    population, train_fitnesses = selection(
        model=model,
        model_full=model,
        processor=processor,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        layers=layers,
        candidates=initial_population_candidates,
        prompt=inputs,
        base_answer=base_answers,
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

        # load_states(model, layers, blocks, population[0])

        # Evaluate current search point
        # if gen_id % args.eval_every == 0 and not args.no_eval:

            # ppl_eval = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)

            # full_train_ppl = compute_clip(model, model_full, clip_model, clip_preprocess, prompt)

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

            if offspring["attn"][0] != False or offspring["attn"][28] != False:
                # print("Error: first block cannot be removed")
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
                model_full=model,
                processor=processor,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                layers=layers,
                candidates=offspring_list,
                prompt=copy.deepcopy(inputs),
                base_answer=base_answers,
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
            with open(os.path.join(args.drop_config_dir, "qwen_layer_skip_config_inference_40_1_only.txt"), "w") as f:
                for line in layer_drop_config:
                    f.write(line + "\n")

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        # Save model
        # torch.save(model, os.path.join(args.save_dir, "final_model.pth"))
        # save_dir1 = os.path.join(args.save_dir, "modified_janus")
        # model.save_pretrained(save_dir1)
        # print("模型已保存至:", save_dir1)

        # Save layer drop config
        with open(os.path.join(args.save_dir, "qwen_layer_drop_config_inference_40_1_only.txt"), "w") as f:
            for line in layer_drop_config:
                f.write(line + "\n")

    print("Final configuration:")
    for line in layer_drop_config:
        print(line)

if __name__ == "__main__":
    main()
