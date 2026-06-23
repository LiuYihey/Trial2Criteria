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
import copy

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

# 保存一份原始配置用于对比
original_ctx_fields = copy.deepcopy(model.config['ctx_fields'])
# 更新模型内部配置 - 完全移除ctx_fields
model.config['ctx_fields'] = []
# 确保BuildModel中的配置也被更新
if hasattr(model, 'model') and hasattr(model.model, 'config'):
    model.model.config['ctx_fields'] = []

print(f"修改前的 ctx_fields: {original_ctx_fields}")
print(f"修改后的 ctx_fields: {model.config['ctx_fields']}")
if hasattr(model, 'model') and hasattr(model.model, 'config'):
    print(f"修改后的 模型内部 ctx_fields: {model.model.config['ctx_fields']}")

# 读取 config，获取 fields
with open(os.path.join(trial2vec_model_path, 'model_config.json'), 'r') as f:
    config = json.load(f)
fields = config['fields']
# 不再使用 ctx_fields (description 和 criteria)

def process_file(model, fields, input_csv_path, output_pth_path):
    """
    读取CSV文件，生成嵌入向量并保存到.pth文件。

    参数:
    - model: 已加载的Trial2Vec模型对象。
    - fields: 用于编码的字段列表。
    - input_csv_path: 输入的CSV文件路径。
    - output_pth_path: 输出的.pth文件路径。
    """
    print(f"\n--- 开始处理文件: {input_csv_path} ---")
    
    # 读取数据
    try:
        df = pd.read_csv(input_csv_path)
        print(f"数据加载成功: {df.shape}")
        print(f"列名: {df.columns.tolist()}")
        print(f"前5行数据:\n{df.head()}")
    except FileNotFoundError:
        print(f"错误: 文件未找到 {input_csv_path}")
        return

    # 构造 inputs 字典，只使用 fields 而不使用 ctx_fields
    inputs = {
        'x': df,
        'fields': fields,
        'ctx_fields': [],  # 使用空列表代替原来的 ctx_fields，不包含 description 和 criteria
        'tag': 'nct_id'
    }

    # 返回字典，key为nct_id，value为embedding向量
    print("开始生成嵌入向量（仅使用fields，不使用description和criteria）...")
    embeddings = model.encode(inputs, return_dict=True)
    print(f"嵌入向量生成完成，共 {len(embeddings)} 个向量")

    # 保存为dict，和官方格式一致
    torch.save({'emb': embeddings}, output_pth_path)
    print(f"嵌入向量已保存到: {output_pth_path}")
    print(f"--- 完成处理文件: {input_csv_path} ---")

# --- 主执行逻辑 ---
# 定义要处理的文件任务列表
tasks = [
    (r'D:\___Desktop___\Trial_project\self\trial_gov\Retrieval_Base\RAG_base_phase3.csv', 'Phase3_emb_only_fields.pth'),
    (r'D:\___Desktop___\Trial_project\self\trial_gov\Retrieval_Base\RAG_base_phase2.csv', 'Phase2_emb_only_fields.pth'),
    (r'D:\___Desktop___\Trial_project\self\trial_gov\Retrieval_Base\RAG_base_phase1.csv', 'Phase1_emb_only_fields.pth')
]

# 依次执行每个任务
for input_path, output_path in tasks:
    process_file(model, fields, input_path, output_path)

print("\n所有任务已完成。")