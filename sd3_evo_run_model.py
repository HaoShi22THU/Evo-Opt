import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from diffusers import StableDiffusion3Pipeline

# # pipe = StableDiffusion3Pipeline.from_pretrained("/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16)
# # pipe = pipe.to("cuda:3")

# # image = pipe(
# #     "Tsinghua University is located in Beijing, China.",
# #     negative_prompt="",
# #     num_inference_steps=28,
# #     guidance_scale=7.0,
# # ).images[0]

# # image.save("output.png")
# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# partial_skip_sd3.py  --  selectively skip self-attention / mlp blocks in SD-3
# """

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"          # ← 改成要用哪张卡

import types
import torch
from diffusers import StableDiffusion3Pipeline
from typing import Iterable, Sequence

# ============================================================
# 1. 打补丁 / 恢复：把 forward 替换为返回 0
# ============================================================

def _replace_forward(mod: torch.nn.Module, tag: str = "") -> None:
    """把子模块的 forward 换成 zeros_like(hidden_states)"""
    if mod is None or hasattr(mod, "__orig_forward"):
        return  # 已经替换或对象为空
    mod.__orig_forward = mod.forward

    def zero_forward(self, hidden_states, *args, **kwargs):
        return torch.zeros_like(hidden_states)

    mod.forward = types.MethodType(zero_forward, mod)
    print(f"→  {tag} patched (zeros)")


def _restore_forward(mod: torch.nn.Module, tag: str = "") -> None:
    """恢复原 forward"""
    if mod is None or not hasattr(mod, "__orig_forward"):
        return
    mod.forward = mod.__orig_forward
    delattr(mod, "__orig_forward")
    print(f"✓  {tag} restored")

# # ============================================================
# # 2. 对 SD-3 transformer_blocks 批量跳过
# # ============================================================

# def patch_sd3_blocks(
#     pipe: StableDiffusion3Pipeline,
#     attn_skip: Sequence[int] = (),
#     mlp_skip: Sequence[int]  = (),
#     restore: bool = False,
# ) -> None:
#     """
#     按层索引跳过 / 恢复 self-attention 与 mlp。

#     Args:
#         pipe:              加载好的 StableDiffusion3Pipeline
#         attn_skip:         需跳过 self-attention 的层索引
#         mlp_skip:          需跳过 feed-forward 的层索引
#         restore:           False=打补丁；True=恢复
#     """
#     blocks = pipe.transformer.transformer_blocks   # nn.ModuleList

#     for idx, blk in enumerate(blocks):
#         # --- self-attention（仅 attn / attn1；保留 attn2） ---
#         if idx in attn_skip:
#             for name in ("attn"):
#                 if hasattr(blk, name):
#                     mod = getattr(blk, name)
#                     (_restore_forward if restore else _replace_forward)(mod, f"{name}[{idx}]")

#         # --- mlp（mlp / mlp1 / mlp2） ---
#         if idx in mlp_skip:
#             for name in ("mlp", "mlp1", "mlp2"):
#                 if hasattr(blk, name):
#                     mod = getattr(blk, name)
#                     (_restore_forward if restore else _replace_forward)(mod, f"{name}[{idx}]")

# # ============================================================
# # 3. DEMO
# # ============================================================

# def main():
#     # ----- 3.1 载入管线 -----
#     model_dir = "/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers"  # ← 改成你的 SD-3 路径
#     pipe = StableDiffusion3Pipeline.from_pretrained(
#         model_dir,
#         torch_dtype=torch.float16
#     ).to("cuda")

#     # ----- 3.2 选择要跳过的层 -----
#     attn_skip = [0, 2, 5]     # 这些层的 self-attn 设为 0
#     mlp_skip  = [1, 4]        # 这些层的 mlp 设为 0

#     # 打补丁
#     patch_sd3_blocks(pipe, attn_skip=attn_skip, mlp_skip=mlp_skip, restore=False)

#     # ----- 3.3 生成图片 -----
#     prompt = "Tsinghua University is located in Beijing, China."
#     image = pipe(
#         prompt,
#         negative_prompt="",
#         num_inference_steps=28,
#         guidance_scale=7.0,
#     ).images[0]

#     os.makedirs("generated_samples", exist_ok=True)
#     out_path = "generated_samples/output_partial_skip.png"
#     image.save(out_path)
#     print(f"Image saved to {out_path}")

#     # ----- 3.4 可选：恢复原网络 -----
#     # patch_sd3_blocks(pipe, attn_skip, mlp_skip, restore=True)

# if __name__ == "__main__":
#     main()

# ---------------------------------------------
# 0. 预备：给管线加一个缓存
# ---------------------------------------------
def _ensure_cache(pipe):
    if not hasattr(pipe, "_last_attn"):
        pipe._last_attn = None

# ---------------------------------------------
# 1. 给“正常执行”的 attn 打包裹 → 先算、再缓存
# ---------------------------------------------
def _wrap_and_cache_attn(attn_mod, pipe, tag=""):
    if hasattr(attn_mod, "__wrapped"):      # 只包一次
        return
    orig_forward = attn_mod.forward

    def wrapped(self, hidden_states, *args, **kwargs):
        out = orig_forward(hidden_states, *args, **kwargs)
        pipe._last_attn = out               # 缓存
        return out

    attn_mod.forward = types.MethodType(wrapped, attn_mod)
    attn_mod.__wrapped = True
    print(f"✓  {tag} wrapped (cache)")

# ---------------------------------------------
# 2. 给“需要复用”的 attn 打补丁 → 直接返回缓存
# ---------------------------------------------
def _reuse_prev_attn(attn_mod, pipe, tag=""):
    if hasattr(attn_mod, "__reused"):
        return
    orig_forward = attn_mod.forward        # 备份一次就够

    def reuse(self, hidden_states, *args, **kwargs):
        # 第一次遇到还没缓存时，退回原计算以避免 None
        if pipe._last_attn is None:
            out = orig_forward(hidden_states, *args, **kwargs)
            pipe._last_attn = out
            return out
        return pipe._last_attn

    attn_mod.forward = types.MethodType(reuse, attn_mod)
    attn_mod.__reused = True
    print(f"→  {tag} patched (reuse prev)")

# ---------------------------------------------
# 3. 主函数：按层批量处理
# ---------------------------------------------
def patch_sd3_blocks_reuse_attn(
    pipe: StableDiffusion3Pipeline,
    attn_reuse: Sequence[int] = (),
    mlp_skip:   Sequence[int] = (),
    restore:    bool = False,
):
    """
    `attn_reuse` 里的层会复用上一层的 attn 输出；
    其它层正常计算但都会把结果写入 pipe._last_attn 作为缓存。
    """
    _ensure_cache(pipe)
    blocks = pipe.transformer.transformer_blocks

    for idx, blk in enumerate(blocks):
        # -------- self-attention --------
        if hasattr(blk, "attn"):            # SD3 把 self-attn 写成 attn
            attn_mod = blk.attn
            tag = f"attn[{idx}]"
            if restore:
                # 还原
                if hasattr(attn_mod, "__orig_forward"):
                    attn_mod.forward = attn_mod.__orig_forward
                    delattr(attn_mod, "__orig_forward")
                    print(f"✓  {tag} restored")
                continue

            if idx in attn_reuse:
                # 复用：改 forward → 直接返回缓存
                if not hasattr(attn_mod, "__orig_forward"):
                    attn_mod.__orig_forward = attn_mod.forward
                _reuse_prev_attn(attn_mod, pipe, tag)
            else:
                # 正常算：包裹一下以写缓存
                if not hasattr(attn_mod, "__orig_forward"):
                    attn_mod.__orig_forward = attn_mod.forward
                _wrap_and_cache_attn(attn_mod, pipe, tag)

        # -------- mlp --------
        # if idx in mlp_skip:
        #     for name in ("mlp", "mlp1", "mlp2"):
        #         if hasattr(blk, name):
        #             mod = getattr(blk, name)
        #             # 复用需求只在 attn，这里仍然用零填充或直接跳过
        #             # 你原来的 _replace_forward / _restore_forward 可以直接复用
        #             (_replace_forward if not restore else _restore_forward)(mod, f"{name}[{idx}]")

# =============================================
# 4. DEMO —— 只修改 main() 里 patch 的调用
# =============================================
def main():
    model_dir = "/mnt/temp/hshi/SD3/stable-diffusion-3-medium-diffusers"
    pipe = StableDiffusion3Pipeline.from_pretrained(
        model_dir, torch_dtype=torch.float16
    ).to("cuda")

    # 加种子
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



    # 想复用上一层 attn 的层（索引从 0 开始）
    attn_reuse = []
    # 仍然想直接跳过的 mlp
    mlp_skip   = []

    patch_sd3_blocks_reuse_attn(pipe, attn_reuse=attn_reuse, mlp_skip=mlp_skip)

    prompt = "Tsinghua University is located in Beijing, China."
    image = pipe(prompt, num_inference_steps=28, guidance_scale=7.0).images[0]
    os.makedirs("generated_samples", exist_ok=True)
    image.save("generated_samples/output_reuse_attn.png")
    # image.save("generated_samples/output_base.png")

    # 恢复（如需要）
    # patch_sd3_blocks_reuse_attn(pipe, restore=True)

if __name__ == "__main__":
    main()