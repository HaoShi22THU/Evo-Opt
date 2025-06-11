# Evo-Opt
  The code for evolution method for LMM. 
 
## Usage

### Download Pre-trained Model:

Our code support 3 mainstream models : [`deepseek-ai/Janus-Pro-7B`](https://huggingface.co/deepseek-ai/Janus-Pro-7B), [`Qwen/Qwen2.5-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct), [`stabilityai/stable-diffusion-3.5-medium`](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium).

### Janus Pro 7B
---
#### Generation

We provide 4 files for the layer skip of the generation of Janus Pro 7B:

* `janus_generation_evo_press_clip.py` - layer skip for the generation process of Janus Pro via clip socre
* `janus_generation_evp_press_ppl.py` - layer skip for the generation process of Janus Pro via complexity
* `janus_generation_run_model.py` - run the skipped Janus Pro via given pattern in generation
* `janus_generation_evo_press_reuse.py` - run the layer reused Janus Pro via given pattern in generation

#### Inference

We provide 2 files for the layer skip of the inference of Janus Pro 7B:

* `janus_inference_evo_press.py` - layer skip for the inference process of Janus Pro via clip socre
* `janus_inference_run_model.py` - run the skipped Janus Pro via given pattern in inference

### Qwen2.5-vl
---
We provide 2 files for the layer skip of the inference of Qwen2.5-vl:

* `qwen_evo_press.py` - layer skip for the inference process of Qwen2.5-vl via clip socre
* `qwen_run_model.py` - run the skipped Qwen2.5-vl via given pattern in inference

### Stable Diffusion 3

We provide 2 files for the layer skip of the inference of Stable Diffusion 3:

* `sd3_evopress_timestep_attn.py` - attn skip for the generation process of Stable Diffusion 3 via clip socre, SSIM and PSNR
* `sd3_run_model.py` - run the skipped Stable Diffusion 3 via given pattern in generation


## Environment

This code was tested on the following environment:
```
pytorch                   2.4.0           py3.10_cuda12.1_cudnn9.1.0_0    pytorch
pytorch-cuda              12.1                 ha16c6d3_5    pytorch
cuda                      12.1.0                        0    nvidia/label/cuda-12.1.0
transformers              4.43.4                   pypi_0    pypi
datasets                  2.21.0                   pypi_0    pypi
lm-eval                   0.4.0                    pypi_0    pypi
```

## Acknowledgment
Our code is based on  [``EvoPress``](https://github.com/IST-DASLab/EvoPress).