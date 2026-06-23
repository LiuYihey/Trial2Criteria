#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import sys

def remove_last_two_columns(input_file, output_file):
    """
    删除CSV文件的最后两列并保存到新文件
    
    Args:
        input_file (str): 输入CSV文件路径
        output_file (str): 输出CSV文件路径
    """
    try:
        print(f"读取文件: {input_file}")
        # 读取CSV文件
        df = pd.read_csv(input_file)
        
        # 获取列数
        num_cols = len(df.columns)
        
        if num_cols < 3:
            print("错误：CSV文件列数少于3，无法删除最后两列")
            return False
            
        # 删除最后两列
        print(f"删除列: {df.columns[-2]}, {df.columns[-1]}")
        df = df.iloc[:, :-2]
        
        # 创建输出目录（如果不存在）
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 保存到新文件
        print(f"保存结果到: {output_file}")
        df.to_csv(output_file, index=False)
        
        print(f"处理完成！从{num_cols}列减少到{len(df.columns)}列")
        return True
    except Exception as e:
        print(f"错误：{str(e)}")
        return False

if __name__ == "__main__":
    # 输入和输出文件路径
    input_file = "drug/drugbank_data_raw.csv"
    output_file = "drug/drugbank_data_v1.csv"
    
    # 执行列删除
    success = remove_last_two_columns(input_file, output_file)
    
    if success:
        print("列删除成功完成")
    else:
        print("列删除失败")
        sys.exit(1) 