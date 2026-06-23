import json 
import csv
import codecs
from collections.abc import Iterable

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

def process_element(element):
    """将单个元素转换为字符串，处理字典和列表"""
    if isinstance(element, dict):
        return json.dumps(element, ensure_ascii=False)
    elif isinstance(element, list):
        return ', '.join(process_element(item) for item in element)
    else:
        return str(element)

# 定义CSV列顺序和字段名
CSV_COLUMNS = [
    'nctId', 'officialTitle', 'statusVerifiedDate','overallStatus', 'phases', 'studyType', 
    'briefSummary', 'detailedDescription', 'conditions', 'keywords',
    'armGroups', 'interventions', 'primaryOutcomes',
    'eligibilityCriteria', 
    'pmid'
]

# 选择是否筛选"completed"状态（True表示筛选，False表示不筛选）
# filter_completed = True
filter_completed = False

# 读取JSON数据
with open('SFT_trials.json', 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

# 统一处理为记录列表
records = json_data if isinstance(json_data, list) else [json_data]

# 转换数据
csv_rows = []
for record in records:
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
    
    # 根据filter_completed选择是否过滤掉"COMPLETED"状态
    if filter_completed and csv_row.get('overallStatus') != 'COMPLETED':
        continue
    
    csv_rows.append(csv_row)

# 创建CSV文件并写入，确保特殊字符正确保存
with open('SFT_trials.csv', 'w', newline='', encoding='utf-8') as csv_file:
    # 添加BOM标记以确保Excel正确识别UTF-8编码
    csv_file.write('\ufeff')
    # 自定义CSV方言以正确处理特殊字符
    csv.register_dialect('escaped', 
                         quoting=csv.QUOTE_ALL,
                         quotechar='"', 
                         doublequote=True,
                         escapechar=None)
    writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS, dialect='escaped')
    writer.writeheader()
    writer.writerows(csv_rows)

print('转换完成。CSV文件已保存，请使用支持UTF-8编码的程序打开。')
