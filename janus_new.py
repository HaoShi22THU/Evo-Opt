import torch
# from transformers import AutoModelForCausalLM
from transformers import AutoConfig, AutoModel


# 加载完整模型（需与训练时结构完全一致）
model_path = "/mnt/temp/hshi/SD3/Janus-Pro-7B"

config = AutoConfig.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_flash_attention_2=True  # 网页1训练参数优化项
)

# 使用通用模型加载接口
full_model = AutoModel.from_pretrained(
    model_path,
    config=config,
    trust_remote_code=True,
    device_map="auto",  # 网页4多卡部署建议
    torch_dtype=torch.bfloat16  # 网页1训练精度说明
)
# full_model = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     trust_remote_code=True,
#     torch_dtype=torch.bfloat16
# ).cuda().eval()

# 提取language_model子模块（基于网页1的模型调用结构）
language_module = full_model.language_model.model

# 保存为独立pth文件（适配网页1的保存方式）
torch.save(language_module.state_dict(), "/mnt/temp/hshi/EvoPress/EvoPress/final_model.pth")

# 验证参数完整性
print(f"模块参数数量：{sum(p.numel() for p in language_module.parameters())}")