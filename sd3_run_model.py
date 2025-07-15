# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# sd3_reuse_attn.py  ——  在 Stable Diffusion 3 的 MM-DiT 里，
#                       让选定 timesteps 复用上一 timestep 的 self-attention 结果
# """

# # ---------------------------------------------------------
# # 0. 运行环境与依赖
# # ---------------------------------------------------------
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"   # ← 改成你想用的 GPU

# import types
# from typing import Sequence

# import torch
# from diffusers import StableDiffusion3Pipeline

# # ---------------------------------------------------------
# # 1. 工具：给管线加缓存、记录当前 timestep
# # ---------------------------------------------------------
# def _ensure_cache(pipe):
#     """初始化 cache 与当前 timestep 记录"""
#     if not hasattr(pipe, "_attn_cache"):
#         pipe._attn_cache = {}      # {block_idx: hidden_states}
#     pipe._cur_step = None


# def patch_mmdit_forward(mmdit, pipe):
#     """
#     包一层 forward：把 timestep 写进 pipe._cur_step
#     只包一次；若已打补丁则直接返回。
#     """
#     if hasattr(mmdit, "__patched_forward"):
#         return

#     mmdit.__orig_forward = mmdit.forward  # 备份

#     def forward_with_step(self, *args, **kwargs):
#         # 1. 找 timestep 参数
#         if "timestep" in kwargs:
#             ts = kwargs["timestep"]
#         elif "timesteps" in kwargs:
#             ts = kwargs["timesteps"]
#         elif len(args) >= 2:
#             ts = args[1]
#         else:
#             raise RuntimeError("forward() 中没找到 timestep(s) 参数")

#         # 2. 转成整数步号（兼容 batch Tensor）
#         if torch.is_tensor(ts):
#             step_int = int(ts.view(-1)[0].item())   # 取第一个元素
#         else:
#             step_int = int(ts)

#         pipe._cur_step = step_int

#         # 3. 调回原始 forward
#         return self.__orig_forward(*args, **kwargs)

#     # 绑定成实例方法
#     mmdit.forward = types.MethodType(forward_with_step, mmdit)
#     mmdit.__patched_forward = True
#     print("✓  MM-DiT.forward wrapped (record timestep)")


# # ---------------------------------------------------------
# # 2. 包装 / 复用每个 block 的 self-attention
# # ---------------------------------------------------------
# def _wrap_attn_for_reuse(attn_mod, pipe, block_idx, reuse_steps):
#     """给单个 self-attention 打补丁，实现复用逻辑"""
#     if hasattr(attn_mod, "__reuse_ready"):
#         return

#     attn_mod.__orig_forward = attn_mod.forward

#     def forward_reuse(self, hidden_states, *args, **kwargs):
#         cur_step = pipe._cur_step

#         # -- 2.1 若在复用步且有缓存 → 直接返回缓存 --
#         if cur_step in reuse_steps and block_idx in pipe._attn_cache:
#             return pipe._attn_cache[block_idx]

#         # -- 2.2 正常计算并写缓存 --
#         out = self.__orig_forward(hidden_states, *args, **kwargs)
#         pipe._attn_cache[block_idx] = out
#         return out

#     attn_mod.forward = types.MethodType(forward_reuse, attn_mod)
#     attn_mod.__reuse_ready = True
#     print(f"→  attn[{block_idx}] wrapped (reuse ready)")


# # ---------------------------------------------------------
# # 3. 对整条 SD-3 管线批量打补丁 / 恢复
# # ---------------------------------------------------------
# def patch_sd3_reuse_attn_by_step(
#     pipe: StableDiffusion3Pipeline,
#     reuse_steps: Sequence[int] = (),
#     restore: bool = False,
# ):
#     """
#     Args:
#         pipe        已加载好的 StableDiffusion3Pipeline
#         reuse_steps 在这些 timestep（int）里，所有 block 的 attn 复用上一 step
#         restore     True 时撤销补丁
#     """
#     _ensure_cache(pipe)

#     # ---- 3.1 定位 MM-DiT 主干 ----
#     mmdit = getattr(pipe, "transformer", None)
#     if mmdit is None:
#         raise AttributeError(
#             "未找到 pipe.transformer；请确认 diffusers 版本或修改属性名"
#         )

#     blocks = mmdit.transformer_blocks  # nn.ModuleList

#     # ---- 3.2 处理 forward（记录 timestep）----
#     if restore:
#         if hasattr(mmdit, "__patched_forward"):
#             mmdit.forward = mmdit.__orig_forward
#             delattr(mmdit, "__patched_forward")
#             print("✓  MM-DiT.forward restored")
#     else:
#         patch_mmdit_forward(mmdit, pipe)

#     # ---- 3.3 遍历 block 的 attn ----
#     for idx, blk in enumerate(blocks):
#         if not hasattr(blk, "attn"):
#             continue
#         attn_mod = blk.attn
#         tag = f"attn[{idx}]"

#         if restore:
#             if hasattr(attn_mod, "__orig_forward"):
#                 attn_mod.forward = attn_mod.__orig_forward
#                 delattr(attn_mod, "__reuse_ready")
#                 print(f"✓  {tag} restored")
#             continue

#         _wrap_attn_for_reuse(attn_mod, pipe, idx, reuse_steps)


# # ---------------------------------------------------------
# # 4. DEMO
# # ---------------------------------------------------------
# def main():

#     # 设置种子
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
    
#     # ---- 4.1 加载 SD-3 ----
#     model_dir = "/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers"   # ←改成自己的路径
#     pipe = StableDiffusion3Pipeline.from_pretrained(
#         model_dir,
#         torch_dtype=torch.float16
#     ).to("cuda")  # 如需多卡/CPU，请自行调整

#     # ---- 4.2 设定要复用的 timesteps ----
#     # 例：偶数 timestep 全部复用，奇数正常计算
#     reuse_steps = [11,12,13,14,15,16,17,18,19,20]   # SD-3 默认 28 步
#     # reuse_steps = []

#     # ---- 4.3 打补丁 ----
#     patch_sd3_reuse_attn_by_step(pipe, reuse_steps=reuse_steps)

#     import time
#     start_time = time.time()
#     # ---- 4.4 生成图片 ----
#     prompt = "Tsinghua University is located in Beijing, China."
#     image = pipe(
#         prompt,
#         negative_prompt="",
#         num_inference_steps=28,
#         guidance_scale=7.0,
#     ).images[0]
#     end_time = time.time()
#     print(f"Generation time: {end_time - start_time:.2f} seconds")

#     os.makedirs("generated_samples", exist_ok=True)
#     out_path = "generated_samples/output_reuse_attn.png"
#     image.save(out_path)
#     print(f"Image saved to {out_path}")

#     # ---- 4.5 如需恢复 ----
#     # patch_sd3_reuse_attn_by_step(pipe, restore=True)


# if __name__ == "__main__":
#     main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
# """
# sd3_reuse_attn_true.py
# ———————————————
# Stable Diffusion 3   (MM-DiT backbone)
# 在指定 timesteps 内复用上一 timestep 的 self-attention 输出，
# 并用计数器 + 计时器验证确已跳过计算。
# """

# import os, types, time
# from typing import Sequence

# import torch
# from diffusers import StableDiffusion3Pipeline


# # -----------------------------------------------------------
# # 1. 实用函数：初始化缓存 & 步号记录
# # -----------------------------------------------------------
# def _ensure_cache(pipe):
#     if not hasattr(pipe, "_attn_cache"):
#         pipe._attn_cache = {}          # {block_idx : tensor}
#     pipe._cur_step = None
#     pipe._cnt = {"compute": 0, "reuse": 0}   # 计数器：真算 / 复用


# # -----------------------------------------------------------
# # 2. 给 MM-DiT.forward 打包：记录 timestep
# # -----------------------------------------------------------
# def patch_mmdit_forward(mmdit, pipe):
#     if hasattr(mmdit, "__patched_forward"):
#         return

#     mmdit.__orig_forward = mmdit.forward

#     def forward_with_step(self, *args, **kwargs):
#         # ------- 找到 timestep 参数 -------
#         if "timestep" in kwargs:
#             ts = kwargs["timestep"]
#         elif "timesteps" in kwargs:
#             ts = kwargs["timesteps"]
#         elif len(args) >= 2:
#             ts = args[1]
#         else:
#             raise RuntimeError("未在 forward() 参数中找到 timestep(s)")

#         if torch.is_tensor(ts):
#             step_int = int(ts.view(-1)[0].item())  # 取第一个元素即可
#         else:
#             step_int = int(ts)
#         pipe._cur_step = step_int
#         return self.__orig_forward(*args, **kwargs)

#     mmdit.forward = types.MethodType(forward_with_step, mmdit)
#     mmdit.__patched_forward = True
#     print("✓  MM-DiT.forward wrapped (record timestep)")


# # -----------------------------------------------------------
# # 3. 给每个 block 的 self-attention 打补丁
# # -----------------------------------------------------------
# def _wrap_attn(attn_mod, pipe, blk_idx, reuse_steps):
#     if hasattr(attn_mod, "__reuse_ready"):
#         return

#     attn_mod.__orig_forward = attn_mod.forward

#     def forward_reuse(self, hidden_states, *a, **kw):
#         cur = pipe._cur_step

#         # ---- 复用分支 ----
#         if cur in reuse_steps and blk_idx in pipe._attn_cache:
#             pipe._cnt["reuse"] += 1
#             return pipe._attn_cache[blk_idx]

#         # ---- 正常计算分支 ----
#         out = self.__orig_forward(hidden_states, *a, **kw)
#         pipe._attn_cache[blk_idx] = out
#         pipe._cnt["compute"] += 1
#         return out

#     attn_mod.forward = types.MethodType(forward_reuse, attn_mod)
#     attn_mod.__reuse_ready = True
#     print(f"→  attn[{blk_idx}] wrapped (reuse-able)")


# # -----------------------------------------------------------
# # 4. 统一接口：打补丁 / 恢复
# # -----------------------------------------------------------
# def patch_sd3_reuse_attn(
#     pipe: StableDiffusion3Pipeline,
#     reuse_steps: Sequence[int],
#     restore: bool = False,
# ):
#     """
#     reuse_steps : e.g. [0,2,4,…]  ——  在这些 timestep 内复用上一步 attn
#     """
#     _ensure_cache(pipe)

#     mmdit = pipe.transformer      # ← 如属性名不同请改这里
#     blocks = mmdit.transformer_blocks

#     # 4-1 处理 backbone.forward
#     if restore:
#         if hasattr(mmdit, "__patched_forward"):
#             mmdit.forward = mmdit.__orig_forward
#             delattr(mmdit, "__patched_forward")
#             print("✓  MM-DiT.forward restored")
#     else:
#         patch_mmdit_forward(mmdit, pipe)

#     # 4-2 处理各 block.attn
#     for idx, blk in enumerate(blocks):
#         if not hasattr(blk, "attn"):
#             continue
#         attn_mod = blk.attn

#         if restore:
#             if hasattr(attn_mod, "__orig_forward"):
#                 attn_mod.forward = attn_mod.__orig_forward
#                 delattr(attn_mod, "__reuse_ready")
#                 print(f"✓  attn[{idx}] restored")
#         else:
#             _wrap_attn(attn_mod, pipe, idx, reuse_steps)


# # -----------------------------------------------------------
# # 5. DEMO & 性能验证
# # -----------------------------------------------------------
# def main():
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"      # ← 改 GPU
#     model_dir = "/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers"  # ← 改路径

#     # ---------- 5.1 载入两份管线（baseline 与 patched） ----------
#     pipe_base = StableDiffusion3Pipeline.from_pretrained(
#         model_dir, torch_dtype=torch.float16
#     ).to("cuda")

#     pipe_fast = StableDiffusion3Pipeline.from_pretrained(
#         model_dir, torch_dtype=torch.float16
#     ).to("cuda")

#     # ---------- 5.2 给 fast 版本打补丁 ----------
#     reuse_steps = [5,6 ,8, 9]   # 偶数步复用
#     patch_sd3_reuse_attn(pipe_fast, reuse_steps=reuse_steps)

#     prompt = "Tsinghua University is located in Beijing, China."

#     # ---------- 5.3 baseline 时间 ----------
#     torch.cuda.synchronize()
#     t0 = time.time()
#     _ = pipe_base(prompt, num_inference_steps=28, guidance_scale=7.0)
#     torch.cuda.synchronize()
#     t_base = time.time() - t0
#     print(f"\n⏱  baseline: {t_base:.2f}s")

#     # ---------- 5.4 patched 时间 ----------
#     torch.cuda.synchronize()
#     t0 = time.time()
#     _ = pipe_fast(prompt, num_inference_steps=28, guidance_scale=7.0)
#     torch.cuda.synchronize()
#     t_fast = time.time() - t0
#     print(f"⏱  reuse-attn: {t_fast:.2f}s")

#     # ---------- 5.5 计数器 ----------
#     print("attention calls → compute:", pipe_fast._cnt["compute"],
#           "| reuse:", pipe_fast._cnt["reuse"])


# if __name__ == "__main__":
#     main()
# ===========================================================
# Stable Diffusion 3 - Attention Reuse (完整示例)
# ===========================================================
#
# 说明
# ----
# • 给定 `reuse_steps`：若当前 timestep ∈ `reuse_steps` → 直接复用
#   上一次「真正计算」得到的各 block-attention；否则重新计算并
#   **覆盖** 缓存（即每遇到新的非复用步，就整体刷新一次缓存）。
# • 计数器 `_cnt = {"compute": …, "reuse": …}` 用来验证是否生效。
# • 如需还原，调用 `patch_sd3_reuse_attn(pipe, …, restore=True)`.
#
# ===========================================================
import os, types, time
from typing import Sequence
os.environ["CUDA_VISIBLE_DEVICES"] = "0"   # ← 改成你想用的 GPU

import torch
from diffusers import StableDiffusion3Pipeline


# -----------------------------------------------------------
# 1. 初始化缓存 & 计数器
# -----------------------------------------------------------
def _ensure_cache(pipe):
    if not hasattr(pipe, "_attn_cache"):
        pipe._attn_cache = {}          # {block_idx: Tensor}
    pipe._cur_step = None              # 当前 timestep（由 forward 捕获）
    pipe._last_compute_step = None     # 上一次真正计算的 timestep
    pipe._cnt = {"compute": 0, "reuse": 0}


# -----------------------------------------------------------
# 2. 包装 MM-DiT.forward：记录 timestep
# -----------------------------------------------------------
def _patch_mmdit_forward(mmdit, pipe):
    if hasattr(mmdit, "__patched_forward"):
        return

    mmdit.__orig_forward = mmdit.forward

    # ---------- 新增：初始化连续计数 ----------
    pipe._step_idx = -1
    pipe._last_ts_int = None

    def forward_with_step(self, *args, **kwargs):
        # ① 抽 timestep -----------------------------------------
        if "timestep" in kwargs:
            ts = kwargs["timestep"]
        elif "timesteps" in kwargs:
            ts = kwargs["timesteps"]
        elif len(args) >= 2:
            ts = args[1]
        else:
            raise RuntimeError("未在 forward() 参数中找到 timestep(s)")

        step_int = int(ts.view(-1)[0].item()) if torch.is_tensor(ts) else int(ts)

        # ② 把“真实噪声步”映射到连续 0‥27 ----------------------
        if step_int != pipe._last_ts_int:          # 进入新的推理 step
            pipe._step_idx += 1
            pipe._last_ts_int = step_int
        pipe._cur_step = pipe._step_idx            ### ← 这里改成连续编号

        return self.__orig_forward(*args, **kwargs)

    mmdit.forward = types.MethodType(forward_with_step, mmdit)
    mmdit.__patched_forward = True
    print("✓  MM-DiT.forward wrapped (record timestep)")
# -----------------------------------------------------------
# 3. 包装每个 block 的 self-attention
# -----------------------------------------------------------
def _wrap_attn(attn_mod, pipe, blk_idx, reuse_steps):
    if hasattr(attn_mod, "__reuse_ready"):
        return                          # 已打补丁，跳过

    attn_mod.__orig_forward = attn_mod.forward

    def forward_reuse(self, hidden_states, *a, **kw):
        cur = pipe._cur_step
        # print(cur)

        # ---- ① 复用分支 ----
        if (cur in reuse_steps) and (blk_idx in pipe._attn_cache):
            pipe._cnt["reuse"] += 1
            return pipe._attn_cache[blk_idx]

        # ---- ② 重新计算分支 ----
        # 若进入新的「真正计算」步 → 整体刷新缓存
        if pipe._last_compute_step != cur:
            pipe._attn_cache.clear()
            pipe._last_compute_step = cur

        out = self.__orig_forward(hidden_states, *a, **kw)
        pipe._attn_cache[blk_idx] = out                # 覆盖 / 写入缓存
        pipe._cnt["compute"] += 1
        return out

    attn_mod.forward = types.MethodType(forward_reuse, attn_mod)
    attn_mod.__reuse_ready = True
    print(f"→  attn[{blk_idx}] wrapped (reuse-able)")


# -----------------------------------------------------------
# 4. 对整条管线打补丁 / 还原
# -----------------------------------------------------------
def patch_sd3_reuse_attn(
    pipe: StableDiffusion3Pipeline,
    reuse_steps: Sequence[int],
    restore: bool = False,
):
    """
    参数
    ----
    pipe         : SD-3 Pipeline
    reuse_steps  : 在这些 timestep 内 *跳过计算*，直接复用上一次结果
    restore      : True → 还原为原始 forward
    """
    _ensure_cache(pipe)

    mmdit = pipe.transformer                       # ← 如属性名不同请改这里
    blocks = mmdit.transformer_blocks

    # 4-1. 处理 backbone.forward
    if restore:
        if hasattr(mmdit, "__patched_forward"):
            mmdit.forward = mmdit.__orig_forward
            delattr(mmdit, "__patched_forward")
            print("✓  MM-DiT.forward restored")
    else:
        _patch_mmdit_forward(mmdit, pipe)

    # 4-2. 处理各 block.attn
    for idx, blk in enumerate(blocks):
        if not hasattr(blk, "attn"):
            continue
        attn_mod = blk.attn

        if restore:
            if hasattr(attn_mod, "__orig_forward"):
                attn_mod.forward = attn_mod.__orig_forward
                delattr(attn_mod, "__reuse_ready")
                print(f"✓  attn[{idx}] restored")
        else:
            _wrap_attn(attn_mod, pipe, idx, reuse_steps)


# -----------------------------------------------------------
# 5. DEMO & 性能验证
# -----------------------------------------------------------
def main():
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True



    os.environ["CUDA_VISIBLE_DEVICES"] = "1"    # ← 改为可用 GPU
    model_dir = "/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers"  # ← 改为模型路径

    gen_base = torch.Generator(device="cuda").manual_seed(42)
    gen_fast = torch.Generator(device="cuda").manual_seed(42)

    # ---------- 5.1 载入两份管线（baseline & patched） ----------
    pipe_base = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=torch.float16
    ).to("cuda")

    pipe_fast = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=torch.float16
    ).to("cuda")

    with open ("/mnt/temp/hshi/EvoPress/EvoPress/timestep_drop_config_70_noclip.txt", "r") as f:
        lines = f.readlines()
    # 读取每一行的内容

    reuse_steps = []
    
    for i in range(len(lines)):
        if lines[i] == "attn\n":
            reuse_steps.append(i)



    # ---------- 5.2 给 fast 版本打补丁 ----------
    # reuse_steps = [i for i in range(11, 25)]                  # 这些步 *复用*
    
    print(reuse_steps)
    patch_sd3_reuse_attn(pipe_fast, reuse_steps=reuse_steps)

    # prompt = "A close-up high-contrast photo of Sydney Opera House sitting next to Eiffel tower, under a blue night sky of roiling energy, exploding yellow stars, and radiating swirls of blue."
    prompt = "lsometric voxel art of a gamer's room withRGB-lit computer setup, posters on the wall, acharacter gaming in a chair, and a cat sitting on the desk by the keyboard."
    # ---------- 5.3 baseline 时间 ----------
    torch.cuda.synchronize()
    t0 = time.time()
    image_base = pipe_base(prompt, num_inference_steps=28, guidance_scale=7.0, generator=gen_base)
    torch.cuda.synchronize()
    t_base = time.time() - t0
    print(f"\n⏱  baseline:   {t_base:.2f}s")

    # ---------- 5.4 patched 时间 ----------
    torch.cuda.synchronize()
    t0 = time.time()
    image_fast = pipe_fast(prompt, num_inference_steps=28, guidance_scale=7.0, generator=gen_fast)
    torch.cuda.synchronize()
    t_fast = time.time() - t0
    print(f"⏱  reuse-attn: {t_fast:.2f}s")

    # ---------- 5.5 保存图片 ----------
    os.makedirs("generated_samples", exist_ok=True)
    out_path_base = "generated_samples/output_base_1.png"
    out_path_fast = "generated_samples/output_fast_70_1.png"

    image_base.images[0].save(out_path_base)
    image_fast.images[0].save(out_path_fast)
    print(f"Image saved to {out_path_base}")
    print(f"Image saved to {out_path_fast}")
    print("✓  images saved")


    # ---------- 5.6 计数器 ----------
    print(reuse_steps)
    print("attention calls → compute:", pipe_fast._cnt["compute"],
          "| reuse:", pipe_fast._cnt["reuse"])


if __name__ == "__main__":
    main()