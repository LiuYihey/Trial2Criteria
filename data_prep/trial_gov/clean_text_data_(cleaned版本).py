#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
清理CSV和JSONL文件中的文本数据:
1. 移除HTML标签
2. 转换Unicode编码和修复转义符问题
"""

import pandas as pd
import re
import json
import argparse
import os
import codecs
from pathlib import Path

def remove_html_tags(text):
    """移除HTML标签如<i></i>"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    clean_text = re.sub(r'<[^>]+>', '', text)
    return clean_text

def unescape_text(text: str) -> str:
    """
    一个健壮的函数，通过迭代解码，来处理所有C风格的转义序列，
    直到字符串稳定为止，从而处理多层嵌套的转义。
    """
    if pd.isna(text) or not isinstance(text, str):
        return text

    # 正则表达式查找任何由反斜杠开头的序列。
    # Group 1: 匹配有效的转义序列 (unicode, hex, octal, 特殊字符)。
    # Group 2: 匹配任何其他在反斜杠之后的字符 (无效转义)。
    robust_escape_pattern = re.compile(r'\\(u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|x[0-9a-fA-F]{2}|[0-7]{1,3}|[\'\"\\abfnrtv])|\\(.)')

    def decode_match_robust(match):
        valid_escape_part = match.group(1)
        invalid_escape_char = match.group(2)

        if valid_escape_part is not None:
            # 这是一个有效的转义序列，重构并解码。
            full_sequence = '\\' + valid_escape_part
            try:
                return codecs.decode(full_sequence, 'unicode_escape')
            except UnicodeDecodeError:
                return full_sequence  # Fallback
        elif invalid_escape_char is not None:
            # 这是一个无效的转义序列 (例如 \>), 直接返回那个字符。
            return invalid_escape_char
        else:
            return match.group(0) # 不应该执行到这里

    # 持续解码，直到字符串不再变化
    previous_text = ""
    while text != previous_text:
        previous_text = text
        text = robust_escape_pattern.sub(decode_match_robust, text)

    return text

def clean_text_base(text):
    """只应用基础清理函数 (HTML, unescape)，不修剪。"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    text = remove_html_tags(text)
    text = unescape_text(text)
    return text

def clean_csv_text(text):
    """为CSV数据应用所有清理函数，包括修剪。"""
    text = clean_text_base(text)
    if pd.isna(text) or not isinstance(text, str):
        return text
    
    # 修剪首尾的空白和引号
    text = text.strip()
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1]
    
    return text

def process_csv(input_file, output_file, columns=None):
    """处理CSV文件中的文本数据"""
    print(f"正在处理CSV文件: {input_file}")
    
    try:
        df = pd.read_csv(input_file, encoding='utf-8')
        print("成功使用编码: utf-8")
        
        original_df = df.copy()

        # 如果没有指定列，处理所有列
        if columns is None:
            columns_to_process = df.select_dtypes(include=['object']).columns
        else:
            columns_to_process = [col for col in columns if col in df.columns]

        print(f"将要处理的列: {', '.join(columns_to_process)}")
        
        # 对指定列应用清理函数
        for col in columns_to_process:
            df[col] = df[col].apply(clean_csv_text)
        
        # 保存处理后的CSV文件
        df.to_csv(output_file, index=False, encoding='utf-8')
        print(f"已保存清理后的CSV文件: {output_file}")
        
        # 显示处理前后的对比
        print("\n处理结果示例:")
        print("-" * 70)
        
        processed_df = df

        for i in original_df.index:
            print(f"--- 行 {i+1} ---")
            for col in columns_to_process:
                original_val = original_df.loc[i, col]
                processed_val = processed_df.loc[i, col]
                if original_val != processed_val:
                    print(f"  列 '{col}':")
                    # 使用repr()来清晰地显示特殊字符如换行符
                    print(f"    处理前: {repr(original_val)}")
                    print(f"    处理后: {repr(processed_val)}")
        print("-" * 70)
        
    except Exception as e:
        print(f"处理CSV文件时出错: {e}")

# ... (process_jsonl and main functions remain largely the same, but will call the new clean_text)

def process_jsonl(input_file, output_file):
    """处理JSONL文件中的文本数据"""
    print(f"正在处理JSONL文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        print("成功使用编码: utf-8")
        
        cleaned_lines = []
        for line in lines:
            try:
                obj = json.loads(line.strip())
                
                def clean_obj(data):
                    if isinstance(data, dict):
                        return {k: clean_obj(v) for k, v in data.items()}
                    elif isinstance(data, list):
                        return [clean_obj(item) for item in data]
                    elif isinstance(data, str):
                        return clean_text_base(data)
                    else:
                        return data
                
                cleaned_obj = clean_obj(obj)
                cleaned_lines.append(json.dumps(cleaned_obj, ensure_ascii=False))
            except json.JSONDecodeError:
                print(f"警告: 无法解析JSON行: {line[:100]}...")
                cleaned_lines.append(line.strip())
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in cleaned_lines:
                f.write(line + '\n')
        
        print(f"已保存清理后的JSONL文件: {output_file}")
        
    except Exception as e:
        print(f"处理JSONL文件时出错: {e}")


def main():
    parser = argparse.ArgumentParser(description='清理CSV和JSONL文件中的文本数据')
    parser.add_argument('input_file', help='输入文件路径 (CSV 或 JSONL)')
    parser.add_argument('-o', '--output', help='输出文件路径 (如果不提供，将自动生成)')
    parser.add_argument('-c', '--columns', nargs='*', help='要处理的CSV列 (空格分隔，不提供则处理所有文本列)')
    args = parser.parse_args()
    
    if not os.path.exists(args.input_file):
        print(f"错误: 找不到输入文件 {args.input_file}")
        return 1
    
    input_path = Path(args.input_file)
    if args.output:
        output_file = args.output
    else:
        output_file = str(input_path.with_name(f"{input_path.stem}_cleaned{input_path.suffix}"))
    
    if output_file == args.input_file:
        backup_file = str(input_path.with_name(f"{input_path.stem}_backup{input_path.suffix}"))
        print(f"正在创建备份: {backup_file}")
        import shutil
        shutil.copy2(args.input_file, backup_file)
    
    if input_path.suffix.lower() == '.csv':
        process_csv(args.input_file, output_file, args.columns)
    elif input_path.suffix.lower() == '.jsonl':
        process_jsonl(args.input_file, output_file)
    else:
        print(f"错误: 不支持的文件类型 {input_path.suffix}")
        return 1
    
    print("\n脚本总结:")
    print("1. 移除了HTML标签。")
    print("2. [迭代式]彻底解码了标准转义序列 (如 \\n -> 换行)。")
    print("3. [迭代式]彻底解码了Unicode编码 (如 \\u03b1 -> α)。")
    print("4. [迭代式]彻底清除了无效转义 (如 \\\\[ND\\\\] -> [ND])。")
    print("5. [仅CSV]移除了每个字段首尾可能存在的多余空格和引号。")
    
    return 0

if __name__ == "__main__":
    exit(main()) 