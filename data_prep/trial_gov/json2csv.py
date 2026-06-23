import json 
import csv
import codecs
from collections.abc import Iterable
from tqdm import tqdm
import html

'''
筛选条件（从SFT_filter_trials.py提取）：
1. overallStatus 为 COMPLETED
2. studyType 为 INTERVENTIONAL
3. officialTitle 不为空
4. eligibilityCriteria 不为空
5. eligibilityCriteria 中包含 Inclusion Criteria 和 Exclusion Criteria
6. 至少有一个 intervention 的 type 为 DRUG
'''

# 筛选条件开关（设置为True表示启用该筛选条件）
filter_completed = True  # 筛选条件1: overallStatus 为 COMPLETED
filter_study_type = True  # 筛选条件2: studyType 为 INTERVENTIONAL
filter_title = True  # 筛选条件3: officialTitle 不为空
filter_eligibility_exists = True  # 筛选条件4: eligibilityCriteria 不为空
filter_eligibility_criteria = True  # 筛选条件5: eligibilityCriteria 中包含 Inclusion Criteria 和 Exclusion Criteria
filter_drug_intervention = True  # 筛选条件6: 至少有一个 intervention 的 type 为 DRUG


def find_values(obj, target_key):
    """递归查找JSON中所有与目标键匹配的值"""
    results = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key == target_key:
                results.append(value)
            results.extend(find_values(value, target_key))
    elif isinstance(obj, list):
        for item in obj:
            results.extend(find_values(item, target_key))
    return results

def has_drug_intervention(record):
    """检查记录中是否至少有一个干预类型为DRUG"""
    # 查找所有interventions字段
    interventions_list = find_values(record, 'interventions')
    
    # 检查每个interventions列表
    for interventions in interventions_list:
        if not isinstance(interventions, list):
            continue
            
        # 检查每个intervention项
        for intervention in interventions:
            if isinstance(intervention, dict) and intervention.get('type', '').upper() == 'DRUG':
                return True
    
    return False

def process_element(element):
    """将单个元素转换为字符串，处理字典和列表"""
    if isinstance(element, dict):
        return json.dumps(element, ensure_ascii=False)
    elif isinstance(element, list):
        return ', '.join(process_element(item) for item in element)
    else:
        # 将元素转换为字符串
        text = str(element)
        # 只处理HTML实体，不替换特殊字符
        text = html.unescape(text)
        return text

def process_special_chars(text):
    """处理HTML实体编码，确保它们在CSV中正确显示"""
    # 只解码HTML实体，保留原始特殊字符
    return html.unescape(text)

# 定义CSV列顺序和字段名
CSV_COLUMNS = [
    'nctId', 'officialTitle', 'overallStatus', 'briefSummary', 'detailedDescription',
    'conditions', 'keywords', 'interventions', 'primaryOutcomes',
    'eligibilityCriteria', 'pmid', 'studyType'  # 添加studyType字段
]

# 读取JSON数据
with open('self/trial_gov/SFT_train_set/SFT_train.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# 统一处理为记录列表
records = json_data if isinstance(json_data, list) else [json_data]

# 转换数据
csv_rows = []
# 添加进度条
for record in tqdm(records, desc="处理记录", unit="条"):
    csv_row = {}
    for column in CSV_COLUMNS:
        # 查找所有匹配字段的值
        found_values = find_values(record, column)
        
        # 处理找到的值
        if not found_values:
            csv_row[column] = 'none'
            continue
        
        # 展开所有可迭代对象（字符串除外）
        all_elements = []
        for value in found_values:
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                if isinstance(value, dict):
                    all_elements.append(value)
                else:
                    all_elements.extend(value)
            else:
                all_elements.append(value)
        
        # 处理每个元素并合并
        processed = [process_element(elem) for elem in all_elements]
        csv_row[column] = ', '.join(processed) if processed else 'none'
    
    # 应用筛选条件
    should_include = True
    
    # 筛选条件1: overallStatus 为 COMPLETED
    if filter_completed and csv_row.get('overallStatus', '').upper() != 'COMPLETED':
        should_include = False
        
    # 筛选条件2: studyType 为 INTERVENTIONAL
    if filter_study_type and csv_row.get('studyType', '').upper() != 'INTERVENTIONAL':
        should_include = False
        
    # 筛选条件3: officialTitle 不为空
    if filter_title and csv_row.get('officialTitle') == 'none':
        should_include = False
        
    # 筛选条件4: eligibilityCriteria 不为空
    if filter_eligibility_exists and csv_row.get('eligibilityCriteria') == 'none':
        should_include = False
        
    # 筛选条件5: eligibilityCriteria 中包含 Inclusion Criteria 和 Exclusion Criteria
    if filter_eligibility_criteria and (
        'Inclusion Criteria' not in csv_row.get('eligibilityCriteria', '') or 
        'Exclusion Criteria' not in csv_row.get('eligibilityCriteria', '')
    ):
        should_include = False
    
    # 筛选条件6: 至少有一个 intervention 的 type 为 DRUG
    if filter_drug_intervention and not has_drug_intervention(record):
        should_include = False
    
    if should_include:
        csv_rows.append(csv_row)

print(f'总记录数: {len(records)}，筛选后记录数: {len(csv_rows)}')

# 创建CSV文件并写入，确保特殊字符正确保存
with open('SFT_train_set.csv', 'w', newline='', encoding='utf-8-sig') as csv_file:
    # 使用utf-8-sig编码自动添加BOM标记
    # 自定义CSV方言以正确处理特殊字符
    csv.register_dialect('escaped', 
                         quoting=csv.QUOTE_ALL,
                         quotechar='"', 
                         doublequote=True,
                         escapechar='\\')  # 添加转义字符
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, dialect='escaped')
    writer.writeheader()
    writer.writerows(csv_rows)

# 计算和打印结果统计
filtered_out_count = len(records) - len(csv_rows)
sample_count = len(csv_rows)  # 样本数量（不包括表头行）

print(f'滤除行数: {filtered_out_count}，占总记录的 {filtered_out_count/len(records)*100:.2f}%')
print(f'最终样本数量: {sample_count} 条（CSV文件中除表头外的行数）')

print('转换完成。CSV文件已保存，请使用支持UTF-8编码的程序打开。')
