import csv
import re
import os
import sys

def analyze_criteria(criteria_text):
    """
    分析criteria内容，检查是否符合筛选条件：
    1. 只包含一个inclusion类和一个exclusion类（允许Key，但总数必须各为1）
    2. 不包含次级分点
    3. 不包含未分点语句/导言
    
    返回：
    - True：符合条件
    - False：不符合条件
    - 不符合条件的原因
    """
    if not criteria_text or not isinstance(criteria_text, str):
        return False, "空内容或非字符串"

    lines = criteria_text.split('\n')

    # 标题正则（允许可选冒号，锚定到8个以上空格开头，兼容Key）
    header_incl_re = re.compile(r'(?im)^\s{8,}(?:Key\s+)?Inclusion\s+Criteria\s*:?\s*$')
    header_excl_re = re.compile(r'(?im)^\s{8,}(?:Key\s+)?Exclusion\s+Criteria\s*:?\s*$')
    # 额外类别（Additional/Appendix/Appendix-Level 等）
    header_additional_re = re.compile(r'(?im)^\s{8,}.*(?:Additional|Appendix(?:-Level)?)\s+(?:Inclusion|Exclusion)\s+Criteria\s*:?\s*$')

    # 顶级分点（10个空格开始）
    top_bullet_re = re.compile(r'^\s{10}(?:\d+\.\s|\*\s|-\s|[a-zA-Z]\.\s|\([a-zA-Z]\)\s)')
    # 次级分点（>=13个空格开始）
    sub_bullet_re = re.compile(r'^\s{13,}(?:\d+\.\s|\*\s|-\s|[a-zA-Z]\.\s|\([a-zA-Z]\)\s)')
    # 导言
    intro_re = re.compile(r'(?i)\b(subjects\s+(must|may\s+not)|participants\s+must)\s*:')

    inclusion_count = len(header_incl_re.findall(criteria_text))
    exclusion_count = len(header_excl_re.findall(criteria_text))
    additional_count = len(header_additional_re.findall(criteria_text))

    if inclusion_count != 1:
        return False, f"inclusion类别数量不等于1（找到{inclusion_count}个）"
    if exclusion_count != 1:
        return False, f"exclusion类别数量不等于1（找到{exclusion_count}个）"
    if additional_count > 0:
        return False, "存在额外的Inclusion/Exclusion类别（如Additional/Appendix）"

    # 在章节内检查：次级分点与未分点语句/导言
    in_inclusion = False
    in_exclusion = False
    for line in lines:
        if header_incl_re.match(line):
            in_inclusion = True
            in_exclusion = False
            continue
        if header_excl_re.match(line):
            in_inclusion = False
            in_exclusion = True
            continue

        if not (in_inclusion or in_exclusion):
            continue

        s = line.rstrip('\r')
        if not s.strip():
            continue

        # 次级分点
        if sub_bullet_re.match(s):
            return False, "包含次级分点"

        # 未分点语句（在章节内出现且有缩进但不是顶级分点），含导言
        if intro_re.search(s):
            return False, "包含导言语句"
        if re.match(r'^\s{10,}', s) and not top_bullet_re.match(s):
            return False, "包含未分点语句"

    # 确认每个章节至少有一个顶级分点
    has_inclusion_items = False
    has_exclusion_items = False
    current = None
    for line in lines:
        if header_incl_re.match(line):
            current = 'incl'
            continue
        if header_excl_re.match(line):
            current = 'excl'
            continue
        if top_bullet_re.match(line):
            if current == 'incl':
                has_inclusion_items = True
            elif current == 'excl':
                has_exclusion_items = True

    if not has_inclusion_items:
        return False, "Inclusion部分没有分点项"
    if not has_exclusion_items:
        return False, "Exclusion部分没有分点项"

    return True, "符合条件"


def unify_bullets_to_star(criteria_text: str) -> str:
    """将顶级分点统一为星号，不改变章节标题与空行。"""
    lines = criteria_text.split('\n')
    out_lines = []
    # 顶级分点匹配并替换为星号
    top_bullet_any_re = re.compile(r'^(\s{10})(?:\d+\.\s|[a-zA-Z]\.\s|\([a-zA-Z]\)\s|-\s|\*\s)(.*)$')
    for line in lines:
        m = top_bullet_any_re.match(line)
        if m:
            content = m.group(2).lstrip()
            out_lines.append(f"{m.group(1)}* {content}")
        else:
            out_lines.append(line)
    return '\n'.join(out_lines)


def filter_csv_file(input_path, output_path):
    """
    读取输入CSV文件，筛选符合条件的行，并写入输出CSV文件
    """
    try:
        # 增加CSV字段大小限制，以处理大的文本字段
        max_int = sys.maxsize
        while True:
            try:
                csv.field_size_limit(max_int)
                break
            except OverflowError:
                max_int = int(max_int / 10)
                
        with open(input_path, 'r', encoding='utf-8-sig') as f_in, \
             open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            
            reader = csv.reader(f_in)
            writer = csv.writer(f_out, quoting=csv.QUOTE_NONNUMERIC)

            header = next(reader)
            writer.writerow(header)
            
            criteria_index = -1
            possible_criteria_columns = ['criteria', 'eligibility_criteria', 'eligibilityCriteria']
            
            for col_name in possible_criteria_columns:
                try:
                    criteria_index = header.index(col_name)
                    print(f"找到criteria列: '{col_name}' (索引: {criteria_index})")
                    break
                except ValueError:
                    continue
            
            if criteria_index == -1:
                print(f"错误: 在{input_path}中没有找到criteria列")
                print(f"可用的列: {header}")
                return
                
            total_count = 0
            filtered_count = 0
            rejection_reasons = {}
            
            for row in reader:
                total_count += 1
                
                if len(row) <= criteria_index:
                    continue
                    
                criteria_text = row[criteria_index]
                is_valid, reason = analyze_criteria(criteria_text)
                
                if is_valid:
                    # 统一分点为星号后写入
                    row[criteria_index] = unify_bullets_to_star(criteria_text)
                    writer.writerow(row)
                    filtered_count += 1
                else:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
            
            print(f"处理完成: 总样本数 {total_count}, 筛选后样本数 {filtered_count}")
            rate = (filtered_count/total_count*100) if total_count else 0.0
            print(f"筛选率: {rate:.2f}%")
            print("拒绝原因统计:")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {reason}: {count} 条")
                
    except FileNotFoundError:
        print(f"错误: 找不到输入文件 '{input_path}'")
    except Exception as e:
        print(f"处理'{input_path}'时发生意外错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    input_file = 'SFT_train_set/SFT_train_set_reformat.csv'
    output_file = 'SFT_train_set/SFT_train_set_filtered.csv'
    
    # 获取绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    
    abs_input_path = os.path.join(project_root, input_file)
    abs_output_path = os.path.join(project_root, output_file)
    
    filter_csv_file(abs_input_path, abs_output_path) 