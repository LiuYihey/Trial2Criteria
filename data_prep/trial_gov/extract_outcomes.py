import csv
import json
# 添加进度条
import tqdm

input_file = 'Retrieval_Base\\Original\\RAG_base_phase1_raw.csv'
output_file = 'Retrieval_Base\\Original\\RAG_base_phase1_outcomes.csv'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8', newline='') as outfile:
    # 获取CSV表头并找到nctId列
    header = next(csv.reader(infile))
    print(f"CSV表头: {header}")
    
    # 查找nctId列
    nct_id_column = None
    for h in header:
        if 'nctid' in h.lower():
            nct_id_column = h
            break
    
    if not nct_id_column:
        print(f"警告：找不到包含'nctId'的列名。使用第一列。")
        nct_id_column = header[0]
    
    print(f"使用列名 '{nct_id_column}' 作为NCT ID")
    
    # 重置文件指针
    infile.seek(0)
    reader = csv.DictReader(infile)
    
    writer = csv.writer(outfile)
    writer.writerow(['nctID', 'outcome_measure'])

    # 计算总行数用于进度条
    with open(input_file, 'r', encoding='utf-8') as count_file:
        csv_reader = csv.reader(count_file)
        next(csv_reader)  # 跳过表头行
        total_rows = sum(1 for _ in csv_reader)  # 只计算数据行
    
    # 重置文件指针
    infile.seek(0)
    reader = csv.DictReader(infile)
    
    # 使用tqdm创建进度条
    progress_bar = tqdm.tqdm(total=total_rows, desc="Processing", unit="rows")
    
    for row in reader:
        # 直接获取nctId列的值
        nctID = row.get(nct_id_column, '')
        
        # 移除可能的引号
        if nctID.startswith('"') and nctID.endswith('"'):
            nctID = nctID[1:-1]
        
        # 处理primaryOutcomes
        primary_outcomes = row.get('primaryOutcomes', '')
        measures = []
        if primary_outcomes:
            try:
                outcomes = json.loads(primary_outcomes)  # 将字符串转换为字典或列表
                # 确保outcomes是列表类型
                if isinstance(outcomes, dict):
                    outcomes = [outcomes]  # 将单个字典转换为包含一个字典的列表
                
                # 提取每个outcome的measure字段
                for idx, outcome in enumerate(outcomes, 1):
                    measure = outcome.get('measure', '').strip()
                    if measure:
                        measures.append(f"{idx}. {measure}.")
            except Exception as e:
                # JSON解析错误时不添加任何内容
                pass
        
        writer.writerow([nctID, ' '.join(measures)])
        progress_bar.update(1)  # 更新进度条
    
    progress_bar.close()  # 关闭进度条

print(f"提取完成，结果已保存到 {output_file}")
