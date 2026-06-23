import pandas as pd

# 读取原始CSV文件的前十行
df = pd.read_csv('SFT_trials.csv', nrows=10)

# 将前十行保存为新的CSV文件
df.to_csv('SFT_trials_first_ten.csv', index=False)

print("已成功提取SFT_trials.csv的前十行并保存为SFT_trials_first_ten.csv") 