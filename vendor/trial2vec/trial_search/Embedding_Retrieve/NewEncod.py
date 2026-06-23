import os
# 设置环境变量，强制使用离线模式
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
# 移除HF镜像设置
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com' 

from trial2vec import Trial2Vec
import pandas as pd
import torch
import json
import sys

# 使用绝对路径指向BERT模型
base_dir = os.path.dirname(os.path.abspath(__file__))
bert_model_path = os.path.join(base_dir, 'pretrained_models/Bio_ClinicalBERT')
trial2vec_model_path = os.path.join(base_dir, 'trial_search/pretrained_trial2vec')

# 检查模型路径是否存在
if not os.path.exists(bert_model_path):
    print(f"错误: BERT模型路径不存在: {bert_model_path}")
    print("请确保模型文件位于正确的位置。")
    sys.exit(1)

if not os.path.exists(trial2vec_model_path):
    print(f"错误: Trial2Vec模型路径不存在: {trial2vec_model_path}")
    print("请确保模型文件位于正确的位置。")
    sys.exit(1)

print(f"使用本地BERT模型: {bert_model_path}")
print(f"使用本地Trial2Vec模型: {trial2vec_model_path}")

try:
    # 加载预训练模型，使用本地BERT模型路径
    model = Trial2Vec(bert_name=bert_model_path)
    model.from_pretrained(trial2vec_model_path)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败: {str(e)}")
    sys.exit(1)

# 读取 config，获取 fields 和 ctx_fields
with open(os.path.join(trial2vec_model_path, 'model_config.json'), 'r') as f:
    config = json.load(f)
fields = config['fields']
ctx_fields = config['ctx_fields']

# 读取数据
df = pd.read_csv(r'C:\Users\86137\Desktop\Trial数据\self\trial_gov\Retrieval_Base\RAG_base_phase1.csv')

print(f"数据加载成功: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"前5行数据:\n{df.head()}")

# 构造 inputs 字典
inputs = {
    'x': df,
    'fields': fields,
    'ctx_fields': ctx_fields,
    'tag': 'nct_id'
}

# 返回字典，key为nct_id，value为embedding向量
print("开始生成嵌入向量...")
embeddings = model.encode(inputs, return_dict=True)
print(f"嵌入向量生成完成，共 {len(embeddings)} 个向量")

# 保存为dict，和官方格式一致
output_path = 'Phase1_emb.pth'
torch.save({'emb': embeddings}, output_path)
print(f"嵌入向量已保存到: {output_path}")