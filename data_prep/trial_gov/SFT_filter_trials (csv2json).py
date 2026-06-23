import csv
import json

'''
筛选条件：
1. overallStatus 为 COMPLETED
2. studyType 为 Interventional
3. officialTitle 不为空
4. eligibilityCriteria 不为空
5. eligibilityCriteria 中包含 Inclusion Criteria 和 Exclusion Criteria
'''

input_file = 'After_date_test_set/test_trials.csv'
output_file = 'After_date_test_trials.json'

count_total = 0
count_excluded = 0
filtered = []

with open(input_file, 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        count_total += 1
        # 检查筛选条件
        if (row.get('overallStatus').upper() == 'COMPLETED' and
            row.get('studyType').upper() == 'INTERVENTIONAL' and
            row.get('officialTitle') and
            row.get('eligibilityCriteria') and
            'Inclusion Criteria' in row['eligibilityCriteria'] and
            'Exclusion Criteria' in row['eligibilityCriteria']):
            filtered.append({
                'officialTitle': row['officialTitle'],
                'eligibilityCriteria': row['eligibilityCriteria']
            })
        else:
            count_excluded += 1

print(f'排除了 {count_excluded} 条，共 {count_total} 条。')

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(filtered, f, ensure_ascii=False, indent=2)

print(f'筛选结果已保存为 {output_file}')