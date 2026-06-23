import re
import json
import os
import time
import pandas as pd

def extract_drug_names(text: str) -> list:
    """
    从输入文本中提取所有 type 为 DRUG 且 name 中不含 'placebo' 的药物名称。
    同时去除剂量信息和通用描述。
    
    参数：
        text: 包含若干 JSON 对象的字符串，例如:
            '{"type": "DRUG", "name": "Diosmin", ...}, {"type": "DRUG", "name": "Placebo", ...}'
    
    返回：
        List[str]: 提取到的净化后的药物名称列表
    """
    # 1. 先把所有单独的 JSON 对象包成一个数组，便于使用 json.loads 解析
    wrapped = f'[{text}]'
    
    try:
        data = json.loads(wrapped)
    except json.JSONDecodeError:
        # 处理无效JSON的情况
        return []
    
    # 2. 过滤出 type 为 DRUG，且 name 中不含 'placebo'（不区分大小写）的条目
    result = []
    for entry in data:
        name = entry.get("name", "")
        if entry.get("type") == "DRUG" and "placebo" not in name.lower():
            # 3. 清理药物名称
            cleaned_name = clean_drug_name(name)
            if cleaned_name:
                result.append(cleaned_name)
    return result

def clean_drug_name(drug_name: str) -> str:
    """
    清理药物名称，去除剂量信息和通用描述。
    
    参数：
        drug_name: 原始药物名称
        
    返回：
        str: 清理后的药物名称
    """
    # 预处理 - 保留百分比溶液的主要成分
    if "%" in drug_name:
        # 尝试提取 "X% Drug" 模式，保留药物名称
        percent_match = re.search(r'(\d+\.?\d*)\s*%\s*([A-Za-z\s\-]+)', drug_name)
        if percent_match:
            # 提取主要药物名称，如从 "5% Dextrose Solution" 提取 "Dextrose"
            drug_name = percent_match.group(2).strip()
    
    # 去除开头的剂量信息，如 "20mg Lisinopril" -> "Lisinopril"
    cleaned = re.sub(r'^\s*\d+(\.\d+)?\s*(mg|mcg|g|ml|mL|μg|IU|IE|units|unit)\s*', '', drug_name, flags=re.IGNORECASE)
    
    # 去除圆括号内的剂量信息，如 (10ml/kg)
    cleaned = re.sub(r'\s*\([^)]*\d[^)]*\)\s*', ' ', cleaned)
    
    # 去除通用描述
    generic_terms = [
        'oral capsule', 'capsule', 'tablet', 'oral tablet', 
        'injection', 'solution', 'suspension', 'fluid expansion',
        'oral solution', 'oral suspension', 'cream', 'ointment',
        'patch', 'suppository', 'spray', 'syrup', 'powder',
        'intravenous', 'intramuscular', 'subcutaneous', 'topical',
        'in water', 'for injection', 'in saline', 'iv'
    ]
    
    for term in generic_terms:
        cleaned = re.sub(rf'\s*{term}\s*', ' ', cleaned, flags=re.IGNORECASE)
    
    # 去除结尾的 mg, mcg, g, ml, 等剂量单位及其数值
    cleaned = re.sub(r'\s+\d+(\.\d+)?\s*(mg|mcg|g|ml|mL|μg|IU|IE|units|unit)\b.*$', '', cleaned, flags=re.IGNORECASE)
    
    # 去除开头的数字和空格
    cleaned = re.sub(r'^\s*\d+(\.\d+)?\s*', '', cleaned)
    
    # 去除常见的缩写标记
    cleaned = re.sub(r'\s*\(([A-Z0-9]+)\)\s*', ' ', cleaned)
    
    # 处理含有'Solution'的情况
    cleaned = re.sub(r'\s+Solution\s+', ' ', cleaned, flags=re.IGNORECASE)
    
    # 去除所有括号及其内容
    cleaned = re.sub(r'\s*\([^)]*\)\s*', ' ', cleaned)
    
    # 去除所有多余空格
    cleaned = re.sub(r'\s+', ' ', cleaned)
    
    # 修剪空白字符
    cleaned = cleaned.strip()
    
    return cleaned

def process_csv_interventions(input_csv, output_csv, batch_size=500):
    """
    处理CSV文件的interventions列，提取每个样本中的药物名称。
    实时保存处理结果，显示进度。
    
    参数:
        input_csv: 输入CSV文件路径
        output_csv: 输出CSV文件路径
        batch_size: 批处理大小，每处理这么多行就保存一次
    """
    print(f"开始处理文件: {input_csv}")
    start_time = time.time()
    
    # 在读取input_csv后，确保output_csv存在（如果不存在则创建）
    if not os.path.exists(output_csv):
        # 创建一个空的DataFrame并写入表头
        pd.DataFrame(columns=['nctId', 'intervention_name']).to_csv(output_csv, index=False, encoding='utf-8')
    
    # 尝试不同的编码读取CSV文件
    encodings = ['utf-8', 'latin1', 'gbk', 'gb2312', 'iso-8859-1']
    
    df = None
    for encoding in encodings:
        try:
            print(f"尝试使用 {encoding} 编码读取文件...")
            df = pd.read_csv(input_csv, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件！")
            break
        except UnicodeDecodeError:
            print(f"{encoding} 编码读取失败，尝试下一个编码...")
        except Exception as e:
            print(f"发生其他错误：{str(e)}")
            raise
    
    if df is None:
        raise ValueError("无法使用任何编码读取CSV文件")
    
    # 确保interventions列存在
    if 'interventions' not in df.columns:
        # 查看列名，寻找可能类似的列
        print(f"CSV文件的列：{df.columns.tolist()}")
        raise ValueError("CSV文件中没有找到'interventions'列")
    
    total_rows = len(df)
    print(f"共有 {total_rows} 条数据需要处理")
    
    # 检查是否有部分已处理的文件
    temp_output = output_csv + ".temp"
    processed_ids = set()
    if os.path.exists(temp_output):
        try:
            temp_df = pd.read_csv(temp_output, encoding='utf-8')
            processed_ids = set(temp_df['nctId'].tolist())
            print(f"发现临时文件，已有 {len(processed_ids)} 条数据处理完成")
        except Exception as e:
            print(f"无法读取临时文件: {str(e)}")
    
    # 创建结果列
    if 'intervention_name' not in df.columns:
        df['intervention_name'] = ""
    
    # 批处理和实时保存
    last_save_time = time.time()
    result_rows = []
    
    for i, row in enumerate(df.itertuples(), 1):
        nct_id = getattr(row, 'nctId')
        
        # 跳过已处理的ID
        if nct_id in processed_ids:
            continue
        
        interventions = getattr(row, 'interventions')
        extracted = []
        
        if pd.notna(interventions):
            extracted = extract_drug_names(str(interventions))
        
        # 更新处理结果
        df.at[row.Index, 'intervention_name'] = ', '.join(extracted) if extracted else ''
        
        # 保存当前处理行供批量保存
        result_rows.append({
            'nctId': nct_id,
            'intervention_name': ', '.join(extracted) if extracted else ''
        })
        
        # 显示进度
        if i % 100 == 0 or i == total_rows:
            elapsed = time.time() - start_time
            percent = (i / total_rows) * 100
            eta = (elapsed / i) * (total_rows - i) if i > 0 else 0
            print(f"进度: {i}/{total_rows} ({percent:.2f}%) - 已用时间: {elapsed:.2f}秒 - 预计剩余: {eta:.2f}秒")
        
        # 定期保存结果
        current_time = time.time()
        if len(result_rows) >= batch_size or (current_time - last_save_time) > 300:  # 每500行或5分钟保存一次
            temp_result_df = pd.DataFrame(result_rows)
            
            # 如果临时文件已存在，追加数据
            if os.path.exists(temp_output):
                temp_result_df.to_csv(temp_output, mode='a', header=False, index=False, encoding='utf-8')
            else:
                temp_result_df.to_csv(temp_output, index=False, encoding='utf-8')
                
            result_rows = []
            processed_ids.update(temp_result_df['nctId'].tolist())
            last_save_time = current_time
            print(f"已保存临时结果，当前已处理 {len(processed_ids)} 条数据")
    
    # 保存最后的批次（如果有）
    if result_rows:
        temp_result_df = pd.DataFrame(result_rows)
        if os.path.exists(temp_output):
            temp_result_df.to_csv(temp_output, mode='a', header=False, index=False, encoding='utf-8')
        else:
            temp_result_df.to_csv(temp_output, index=False, encoding='utf-8')
    
    # 读取完整的临时文件并保存为最终结果
    if os.path.exists(temp_output):
        final_df = pd.read_csv(temp_output, encoding='utf-8')
        final_df.to_csv(output_csv, index=False, encoding='utf-8')
        print(f"所有数据处理完成，结果已保存到 {output_csv}")
    else:
        # 直接保存当前处理的结果
        df[['nctId', 'intervention_name']].to_csv(output_csv, index=False, encoding='utf-8')
        print(f"所有数据处理完成，结果已保存到 {output_csv}")
    
    # 处理完成后可以删除临时文件
    try:
        if os.path.exists(temp_output):
            os.remove(temp_output)
            print("临时文件已删除")
    except Exception as e:
        print(f"无法删除临时文件: {str(e)}")
    
    total_elapsed = time.time() - start_time
    print(f"总处理时间: {total_elapsed:.2f}秒")
