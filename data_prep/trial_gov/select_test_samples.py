import pandas as pd
import numpy as np
import re

# 设置随机种子以确保结果可重复
np.random.seed(42)

# 读取原始CSV文件
print("正在读取SFT_trials.csv...")
df = pd.read_csv('SFT_trials.csv')
total_count = len(df)
print(f"总共读取了 {total_count} 条记录")

# 筛选满足条件的样本
print("正在筛选满足条件的样本...")
filtered_df = df[
    (df['overallStatus'] == 'COMPLETED') & 
    (df['phases'] == 'PHASE3') & 
    (df['studyType'] == 'INTERVENTIONAL') & 
    (df['interventions'].str.contains('DRUG', na=False, case=True, regex=True))
]
filtered_count = len(filtered_df)
print(f"筛选出满足条件的记录：{filtered_count} 条")

if filtered_count < 787:
    print(f"警告：满足条件的样本数量({filtered_count})少于要求的787例")
    test_samples_count = filtered_count
else:
    test_samples_count = 787

# 随机选择787例（或全部满足条件的样本，如果少于787）
# 获取索引并随机打乱
indices = filtered_df.index.tolist()
np.random.shuffle(indices)
test_indices = indices[:test_samples_count]

# 分离测试集和训练集
test_df = df.loc[test_indices]
train_df = df.drop(test_indices)

print(f"测试集样本数：{len(test_df)}")
print(f"训练集样本数：{len(train_df)}")

# 保存为CSV文件
test_df.to_csv('SFT_test_samples.csv', index=False)
train_df.to_csv('SFT_train_samples.csv', index=False)

print("已完成：")
print(f"- 测试样本已保存到 SFT_test_samples.csv（{len(test_df)} 条记录）")
print(f"- 训练样本已保存到 SFT_train_samples.csv（{len(train_df)} 条记录）") 