import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from trial2vec import Trial2Vec
import pandas as pd
import torch
import json

# 加载预训练模型（会自动读取config和权重）
model = Trial2Vec()
model.from_pretrained('./trial_search/pretrained_trial2vec')

# 读取 config，获取 fields 和 ctx_fields
with open('./trial_search/pretrained_trial2vec/model_config.json', 'r') as f:
    config = json.load(f)
fields = config['fields']
ctx_fields = config['ctx_fields']

# 读取数据
df = pd.read_csv(r'C:\Users\86137\Desktop\Trial_project\self\trial_gov\Retrieval_Base\RAG_base_phase3.csv')

print(f"原始df行数: {len(df)}")
print(f"df字段: {list(df.columns)}")

# 检查每个fields/ctx_fields的缺失比例
def col_missing(col):
    if col in df.columns:
        miss = df[col].isna().sum()
        print(f"{col}: 缺失 {miss} / {len(df)} ({miss/len(df)*100:.2f}%)")
    else:
        print(f"{col}: 不在df中！")
for col in fields + ctx_fields:
    col_missing(col)

# 检查每行是否全为空（主字段+ctx字段）
def is_all_empty(row):
    return all((pd.isna(row[f]) or str(row[f]).strip() == '' or str(row[f]).lower() == 'none') for f in fields+ctx_fields if f in row)
empty_rows = df.apply(is_all_empty, axis=1)
print(f"所有字段全空的trial数: {empty_rows.sum()} / {len(df)}")

# 构造 inputs 字典
inputs = {
    'x': df,
    'fields': fields,
    'ctx_fields': ctx_fields,
    'tag': 'nct_id'
}

# 检查 dataloader 长度和每个batch内容
try:
    dataloader = model.get_val_dataloader(inputs)
    print(f"dataloader长度（batch数）: {len(dataloader)}")
    for i, batch in enumerate(dataloader):
        tags = batch['nct_id'] if 'nct_id' in batch else None
        print(f"Batch {i+1}: trial数={len(tags) if tags is not None else '未知'}，tag样例={tags[:5] if tags is not None else '无'}")
        if i >= 2:
            break
except Exception as e:
    print(f"get_val_dataloader出错: {e}")

# 输出被过滤的trial示例
if empty_rows.sum() > 0:
    print("被过滤的trial示例（全空）:")
    print(df[empty_rows].head())

# 暂不执行encode，待调试输出后再决定
# embeddings = model.encode(inputs, return_dict=True)
# torch.save({'emb': embeddings}, 'Phase3_emb.pth')