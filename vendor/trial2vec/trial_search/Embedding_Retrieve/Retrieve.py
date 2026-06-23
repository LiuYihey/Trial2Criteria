import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# 添加离线模式标志
os.environ['HF_HUB_OFFLINE'] = '1'
import torch
from torch.serialization import add_safe_globals
import pandas as pd
import json
from trial2vec import Trial2Vec

# 添加numpy安全全局设置
add_safe_globals(['numpy.core.multiarray._reconstruct'])

# 1. 加载 Phase3_emb.pth 和原始csv
emb_dict = torch.load('self/trial_gov/Trial2Vec-main/Phase3_emb.pth', weights_only=False)['emb']
all_ids = list(emb_dict.keys())
all_embs = list(emb_dict.values())

# 使用相对路径
raw_df = pd.read_csv('self/trial_gov/Retrieval_Base/RAG_base_phase3.csv')
raw_df = raw_df.set_index('nct_id')

# 2. 加载 Trial2Vec 模型和权重
# model = Trial2Vec()
model = Trial2Vec(bert_name='self/trial_gov/Trial2Vec-main/pretrained_models/Bio_ClinicalBERT')
model.from_pretrained('self/trial_gov/Trial2Vec-main/trial_search/pretrained_trial2vec')

# 3. 读取 demo case
case_df = pd.read_csv('self/trial_gov/Retrieval_Base/RAG_base_phase3.csv')
query_row = case_df[case_df['nct_id'] == 'NCT02216123']
if query_row.empty:
    raise ValueError('{} not found in demo_data/clinical_trial_mini.csv'.format(query_row['nct_id']))

# 4. 获取 config 字段
with open('self/trial_gov/Trial2Vec-main/trial_search/pretrained_trial2vec/model_config.json', 'r') as f:
    config = json.load(f)
fields = config['fields']
ctx_fields = config['ctx_fields']

# 5. 对 case 进行 encoding
inputs = {
    'x': query_row,
    'fields': fields,
    'ctx_fields': ctx_fields,
    'tag': 'nct_id'
}
query_emb_dict = model.encode(inputs, return_dict=True)
query_emb = list(query_emb_dict.values())[0]  # shape: (dim,)

# 6. 计算相似度（余弦）
import numpy as np
# 定义归一化函数，将向量除以其L2范数，使其长度为1
def normalize(x): return x / np.linalg.norm(x, axis=-1, keepdims=True)
# 将查询向量reshape为(1, dim)并归一化
query_emb_norm = normalize(query_emb.reshape(1, -1))
# 将所有向量堆叠成矩阵并归一化
all_embs_norm = normalize(np.stack(all_embs))
# 计算余弦相似度：点积两个归一化向量，并将结果reshape为一维数组
sims = np.dot(all_embs_norm, query_emb_norm.T).reshape(-1)
# 获取相似度从高到低的索引
topk_idx = np.argsort(sims)[::-1]
# 根据索引获取对应的trial ID
topk_ids = [all_ids[i] for i in topk_idx]
# 获取排序后的相似度值
topk_sims = sims[topk_idx]

# 7. 查找原始内容并输出 top10
# 从原始数据中提取前10个最相似的trial数据
result_df = raw_df.loc[topk_ids[:10]].copy()
# 添加相似度列到结果DataFrame
result_df['similarity'] = topk_sims[:10]
# 重置索引并打印结果
print(result_df.reset_index())
