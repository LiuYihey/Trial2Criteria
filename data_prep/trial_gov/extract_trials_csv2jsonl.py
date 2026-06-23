#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提取CSV文件中的officialTitle和eligibilityCriteria字段，并创建JSONL文件
使用pandas库处理CSV文件
"""

import pandas as pd
import json
import os
import sys


def process_csv(input_file, output_file):
    """处理CSV文件并创建JSONL文件"""
    try:
        # 使用pandas读取CSV文件
        print(f"正在读取CSV文件: {input_file}")
        df = pd.read_csv(input_file, encoding='utf-8')
        
        # 检查必要的列是否存在
        required_columns = ['nctId', 'officialTitle', 'eligibilityCriteria']
        for col in required_columns:
            if col not in df.columns:
                print(f"错误: CSV文件中缺少'{col}'列")
                print(f"可用的列: {df.columns.tolist()}")
                sys.exit(1)
        
        # 打开输出文件
        with open(output_file, 'w', encoding='utf-8') as jsonlfile:
            # 计数器
            count = 0
            
            # 处理每一行
            for _, row in df.iterrows():
                try:
                    # 提取需要的字段，直接转换为字符串并处理空值
                    nct_id = "" if pd.isna(row['nctId']) else str(row['nctId'])
                    official_title = "" if pd.isna(row['officialTitle']) else str(row['officialTitle'])
                    eligibility_criteria = "" if pd.isna(row['eligibilityCriteria']) else str(row['eligibilityCriteria'])
                    
                    # 创建JSON对象
                    json_obj = {
                        'nctId': nct_id,
                        'officialTitle': official_title,
                        'eligibilityCriteria': eligibility_criteria
                    }
                    
                    # 写入JSONL文件
                    jsonlfile.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
                    
                    count += 1
                    if count % 100 == 0:
                        print(f"已处理 {count} 条记录")
                        
                except Exception as e:
                    print(f"处理记录时出错: {e}")
                    continue
            
            print(f"处理完成，共处理 {count} 条记录")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")
        sys.exit(1)

def main():
    # 定义输入和输出文件路径
    input_file = 'self/trial_gov/SFT_train_test/SFT_train_set_831.csv'
    output_file = 'self/trial_gov/SFT_train_test/SFT_train_set_831.jsonl'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 处理CSV文件
    process_csv(input_file, output_file)

if __name__ == "__main__":
    main() 