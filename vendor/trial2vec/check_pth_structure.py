import os
import torch

# 先自动下载预训练模型
# try:
#     from trial2vec import download_embedding
#     print("Downloading pretrained model (if not already present)...")
#     download_embedding()
# except Exception as e:
#     print(f"Warning: download_embedding failed: {e}")

# 默认路径
MODEL_DIR = './trial_search/pretrained_trial2vec'

# 自动查找pth或pth.tar文件
pth_file = None
if os.path.isdir(MODEL_DIR):
    for f in os.listdir(MODEL_DIR):
        if f.endswith('.pth') or f.endswith('.pth.tar'):
            pth_file = os.path.join(MODEL_DIR, f)
            break

if pth_file is None:
    raise FileNotFoundError(f'No .pth or .pth.tar file found in {MODEL_DIR}')

print(f'Loading: {pth_file}')
state = torch.load(pth_file, map_location='cpu', weights_only=False)

print('Top-level keys:')
for k in state:
    v = state[k]
    if isinstance(v, dict):
        print(f'  {k}: dict, {len(v)} keys')
    elif isinstance(v, (list, tuple)):
        print(f'  {k}: {type(v).__name__}, len={len(v)}')
    elif hasattr(v, 'shape'):
        print(f'  {k}: {type(v).__name__}, shape={v.shape}')
    else:
        print(f'  {k}: {type(v).__name__}')

if "emb" in state:
    emb = state["emb"]
    print(f"\n'emb' key found. Type: {type(emb)}, len: {len(emb)}")
    # 打印前3个trial的embedding信息
    for i, (trial_id, emb_val) in enumerate(emb.items()):
        print(f"  trial_id: {trial_id}, embedding shape: {getattr(emb_val, 'shape', type(emb_val))}")
        if i >= 2:
            break
else:
    print("\nNo 'emb' key found in the state dict.") 