"""
下载 Time-MoE 模型到本地文件夹
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 使用镜像加速

from huggingface_hub import snapshot_download

# 模型保存路径
LOCAL_DIR = r"D:\Junshan_Project\models\time-moe-200m"

print("⏳ 开始下载 TimeMoE-200M 模型...")
print(f"   保存到: {LOCAL_DIR}")

# 下载模型
snapshot_download(
    repo_id="Maple728/TimeMoE-200M",
    local_dir=LOCAL_DIR,
    resume_download=True,
)

print("✅ 模型下载完成！")
print(f"   模型位置: {LOCAL_DIR}")
