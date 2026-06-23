import pandas as pd
import os
import argparse
import re
import unicodedata
import csv  # 引入csv模块以使用其常量

'''
详细说明：
1. 合并文件
2. 重命名列
3. 删除列
4. 保存结果
'''
def clean_text(text):
    """
    清理文本中的问题字符，但完全保留换行、缩进和所有空格格式
    """
    if not isinstance(text, str):
        return text
    
    # 移除控制字符，但保留换行符、制表符和空格
    text = ''.join(ch for ch in text if unicodedata.category(ch)[0] != 'C' or ch in ['\n', '\t', ' '])
    
    # 替换常见问题字符
    text = text.replace('â', '-')  # 常见的破折号错误编码
    text = text.replace('â\x80\x99', "'")  # 单引号错误编码
    text = text.replace('â\x80\x93', "-")  # 破折号错误编码
    text = text.replace('â\x89¥', "≥")   # 大于等于符号
    text = text.replace('â\x89¤', "≤")   # 小于等于符号
    
    # 不处理多个空格，完全保留原始格式
    
    # 处理转义符号，例如 \\< -> <, \< -> <, \\> -> >, \> -> > 等
    # 递归处理所有的转义字符
    while True:
        new_text = re.sub(r'\\(.)', r'\1', text)
        if new_text == text:  # 如果没有变化，说明已经处理完所有转义字符
            break
        text = new_text
    
    return text

def clean_dataframe(df):
    """清理DataFrame中所有字符串列的文本"""
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].apply(clean_text)
    return df

def merge_csv_files(base_file, files_to_merge, id_column='nctId'):
    """根据指定的ID列合并多个CSV文件"""
    # 加载基础文件
    print(f"读取基础文件: {base_file}")
    try:
        # 尝试使用UTF-8编码，确保正确处理引号和换行符
        # QUOTE_ALL确保每个字段都被引号包围，这有助于保留格式
        base_df = pd.read_csv(base_file, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
    except UnicodeDecodeError:
        try:
            print("UTF-8编码读取失败，尝试使用UTF-8-SIG编码(带BOM)")
            base_df = pd.read_csv(base_file, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        except UnicodeDecodeError:
            print("UTF-8-SIG编码读取失败，尝试使用latin1编码")
            base_df = pd.read_csv(base_file, encoding='latin1', quoting=csv.QUOTE_ALL)
    
    # 清理基础文件
    base_df = clean_dataframe(base_df)
    
    # 处理列名，确保没有BOM标记
    base_df.columns = [col.replace('ï»¿', '').strip('"') for col in base_df.columns]
    
    # 调整nctId/nctID列名统一性
    if 'nctID' in base_df.columns and id_column == 'nctId':
        print("将'nctID'列重命名为'nctId'")
        base_df = base_df.rename(columns={'nctID': 'nctId'})
    elif 'nctId' in base_df.columns and id_column == 'nctID':
        print("将'nctId'列重命名为'nctID'")
        base_df = base_df.rename(columns={'nctId': 'nctID'})
    
    # 逐个合并其他文件
    for file_path in files_to_merge:
        if not os.path.exists(file_path):
            print(f"警告: 文件 {file_path} 不存在，已跳过")
            continue
            
        print(f"合并文件: {file_path}")
        try:
            # 尝试使用UTF-8编码，确保正确处理引号和换行符
            df = pd.read_csv(file_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
        except UnicodeDecodeError:
            try:
                print(f"UTF-8编码读取失败，尝试使用UTF-8-SIG编码(带BOM)")
                df = pd.read_csv(file_path, encoding='utf-8-sig', quoting=csv.QUOTE_ALL)
            except UnicodeDecodeError:
                print(f"UTF-8-SIG编码读取失败，尝试使用latin1编码")
                df = pd.read_csv(file_path, encoding='latin1', quoting=csv.QUOTE_ALL)
        
        # 清理数据
        df = clean_dataframe(df)
        
        # 处理列名，确保没有BOM标记
        df.columns = [col.replace('ï»¿', '').strip('"') for col in df.columns]
        
        # 调整nctId/nctID列名统一性
        if 'nctID' in df.columns and id_column == 'nctId':
            print("将文件中的'nctID'列重命名为'nctId'")
            df = df.rename(columns={'nctID': 'nctId'})
        elif 'nctId' in df.columns and id_column == 'nctID':
            print("将文件中的'nctId'列重命名为'nctID'")
            df = df.rename(columns={'nctId': 'nctID'})
        
        # 确保合并列存在
        if id_column not in base_df.columns:
            print(f"错误: 基础文件中没有列 '{id_column}'")
            print(f"可用的列: {list(base_df.columns)}")
            return None
            
        if id_column not in df.columns:
            print(f"警告: 文件 {file_path} 中没有列 '{id_column}'，已跳过")
            print(f"可用的列: {list(df.columns)}")
            continue
            
        # 合并文件
        base_df = base_df.merge(df, on=id_column, how='left')
        print(f"  - 合并后的列: {list(base_df.columns)}")
    
    return base_df

def rename_columns(df, rename_dict=None, rename_by_index=None):
    """重命名DataFrame中的列，支持按名称或索引重命名"""
    if rename_dict:
        # 按名称重命名
        missing_cols = [col for col in rename_dict.keys() if col not in df.columns]
        if missing_cols:
            print(f"警告: 以下列不存在，无法重命名: {missing_cols}")
            
        # 重命名列
        df = df.rename(columns=rename_dict)
        print(f"已重命名列: {list(rename_dict.items())}")
    
    if rename_by_index:
        # 按索引重命名
        current_columns = list(df.columns)
        for idx, new_name in rename_by_index.items():
            if idx < len(current_columns):
                old_name = current_columns[idx]
                current_columns[idx] = new_name
                print(f"  - 将列 {idx}('{old_name}') 重命名为 '{new_name}'")
            else:
                print(f"警告: 索引 {idx} 超出范围，无法重命名")
        
        df.columns = current_columns
    
    return df

def drop_columns(df, columns_to_drop):
    """删除DataFrame中的指定列"""
    # 检查要删除的列是否存在
    existing_cols = [col for col in columns_to_drop if col in df.columns]
    missing_cols = [col for col in columns_to_drop if col not in df.columns]
    
    if missing_cols:
        print(f"警告: 以下列不存在，无法删除: {missing_cols}")
    
    if existing_cols:
        df = df.drop(columns=existing_cols)
        print(f"已删除列: {existing_cols}")
    
    return df

def save_csv(df, output_file):
    """保存DataFrame到CSV文件，确保保留所有格式并正确处理编码"""
    # 确保输出目录存在
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    
    # 使用utf-8-sig保存(包含BOM)，确保所有字段都被引号包围，正确处理换行和特殊字符
    df.to_csv(output_file, index=False, encoding='utf-8-sig', 
              quoting=csv.QUOTE_ALL,        # 确保所有字段都被引号包围
              quotechar='"',                # 使用双引号作为引号字符
              doublequote=True,             # 使用双引号来处理引号内的引号，不需要额外的转义字符
              lineterminator='\n')          # 修正：使用lineterminator而不是line_terminator
    
    print(f"已保存CSV文件: {output_file}")
    print(f"文件包含 {len(df)} 行数据和 {len(df.columns)} 列")
    print(f"列名: {list(df.columns)}")

def main():
    # 1. 设置输入输出文件路径
    base_file = 'Retrieval_Base\\Original\\RAG_base_phase2_raw.csv'
    files_to_merge = [
        'Retrieval_Base\\Original\\RAG_base_phase2_outcomes.csv',
        'Retrieval_Base\\Original\\extracted_drugs_phase2.csv'
    ]
    output_file = 'Retrieval_Base\\Original\\RAG_base_phase2.csv'
    
    # 2. 合并文件
    print("=== 步骤1: 合并CSV文件 ===")
    merged_df = merge_csv_files(base_file, files_to_merge)
    if merged_df is None:
        print("合并失败，程序终止")
        return
    
    # 打印当前列名
    print("\n当前列名:")
    for i, col in enumerate(merged_df.columns):
        print(f"{i}: {col}")
    
    # 3. 重命名列
    print("\n=== 步骤2: 重命名列 ===")
    # 示例1: 按名称重命名
    column_mapping = {
        'officialTitle': 'title',
        'briefSummary': 'description',
        'nctId': 'nct_id',
        'conditions': 'disease',
        'keywords': 'keyword',
        'eligibilityCriteria': 'criteria'
    }
    
    # # 示例2: 按索引位置重命名
    # column_index_mapping = {
    #     0: 'trial_id',  # 将第一列重命名为trial_id
    #     4: 'condition'   # 将第五列重命名为condition
    # }
    
    # 应用重命名
    merged_df = rename_columns(merged_df, rename_dict=column_mapping, rename_by_index=None)
    
    # 4. 删除列
    print("\n=== 步骤3: 删除列 ===")
    columns_to_drop = ['pmid', 'primaryOutcomes', 'interventions', 'overallStatus']  # 示例: 删除pmid列
    merged_df = drop_columns(merged_df, columns_to_drop)
    
    # 5. 保存结果
    print("\n=== 步骤4: 保存结果 ===")
    save_csv(merged_df, output_file)
    
    print("\n处理完成!")

if __name__ == "__main__":
    main() 