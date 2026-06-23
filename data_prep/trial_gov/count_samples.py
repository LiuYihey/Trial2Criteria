import pandas as pd
import os

print("开始计算SFT_trials.csv的样本数量...")

# 获取文件大小
file_size = os.path.getsize('SFT_trials.csv')
print(f"文件大小: {file_size/1024/1024:.2f} MB")

# 方法1：使用pandas读取并计算行数
try:
    print("正在使用pandas读取文件...")
    df = pd.read_csv('SFT_trials.csv')
    print(f"CSV文件中的样本数量（不含标题行）: {len(df)}")
    
    # 查看一些基本信息
    print("\n数据集基本信息:")
    print(f"列数: {len(df.columns)}")
    print(f"列名: {list(df.columns)}")
except Exception as e:
    print(f"使用pandas读取文件时出错: {e}")
    
    # 方法2：使用基本文件操作逐行计数
    try:
        print("\n尝试使用基本文件读取方式计数...")
        with open('SFT_trials.csv', 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        print(f"CSV文件总行数（含标题行）: {line_count}")
        print(f"CSV文件样本数量（不含标题行）: {line_count - 1}")
    except Exception as e2:
        print(f"使用基本文件读取方式计数时出错: {e2}") 