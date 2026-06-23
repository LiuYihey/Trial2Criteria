import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download

model_name = "emilyalsentzer/Bio_ClinicalBERT"
local_dir = "./pretrained_models/Bio_ClinicalBERT"

# 直接下载到指定目录（不是cache_dir）
snapshot_download(
    repo_id="emilyalsentzer/Bio_ClinicalBERT",
    local_dir="./pretrained_models/Bio_ClinicalBERT",
    local_dir_use_symlinks=False
)

print("模型和分词器已下载到", local_dir)