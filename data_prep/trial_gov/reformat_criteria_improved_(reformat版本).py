import csv
import os
import re
import sys

def pre_clean_text(text: str) -> str:
    """
    只执行基本的文本清理，不修改结构
    - 递归处理转义字符
    - 保留所有换行和空格
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 处理转义字符
    clean_text = text
    while True:
        new_text = re.sub(r'\\(.)', r'\1', clean_text)
        if new_text == clean_text:
            break
        clean_text = new_text
    
    return clean_text

def normalize_criteria_structure(text: str, verbose=False) -> str:
    """
    标准化criteria格式，使其与clinical_trials.csv保持一致
    - 保留所有原始内容
    - 严格识别带冒号的标题
    - 识别导言模式
    - 处理分点类型变化导致的层级变化
    - 递归处理多级缩进
    - 精确控制换行
    """
    if verbose:
        print(f"输入文本 ({len(text)} 字符):\n{text[:100]}...")
    
    clean_text = pre_clean_text(text)
    if not clean_text:
        return ""
    
    # 如果没有任何标题标记，自动添加标准标题
    if not re.search(r'(?i)(inclusion|exclusion)\s+criteria\s*:', clean_text):
        # 检查是否包含星号项目
        if re.search(r'^\s*\*\s', clean_text, re.MULTILINE):
            # 有星号项目但没有标题，添加默认标题
            if verbose:
                print("未发现标题，但检测到星号项目，添加默认标题")
            
            # 查找第一个星号的位置
            first_star = re.search(r'^\s*\*\s', clean_text, re.MULTILINE)
            if first_star:
                # 在第一个星号前插入标题
                pos = first_star.start()
                clean_text = clean_text[:pos] + "Inclusion Criteria:\n\n" + clean_text[pos:]
    
    # 严格匹配带冒号的标题
    clean_text = re.sub(r'(?i)key\s+inclusion\s+criteria\s*:', 'KEY_INCLUSIONCRITERIA:', clean_text)
    clean_text = re.sub(r'(?i)key\s+exclusion\s+criteria\s*:', 'KEY_EXCLUSIONCRITERIA:', clean_text)
    clean_text = re.sub(r'(?i)inclusion\s+criteria\s*:', 'INCLUSIONCRITERIA:', clean_text)
    clean_text = re.sub(r'(?i)exclusion\s+criteria\s*:', 'EXCLUSIONCRITERIA:', clean_text)
    
    parts = re.split(r'(KEY_INCLUSIONCRITERIA:|KEY_EXCLUSIONCRITERIA:|INCLUSIONCRITERIA:|EXCLUSIONCRITERIA:)', clean_text)
    
    if verbose:
        print(f"分割后部分数量: {len(parts)}")
        for i, part in enumerate(parts):
            print(f"Part {i}: {part[:50]}...")
    
    result = []
    result.append("")
    
    # 如果没有找到任何标题部分，处理整个文本作为一个部分
    if len(parts) == 1:
        if verbose:
            print("未找到标题标记，处理整个文本")
            
        content_text = parts[0].strip()
        if content_text:
            result.append("        Inclusion Criteria:")
            result.append("")
            
            # 分析和格式化内容
            formatted_content = format_content_section(content_text, verbose)
            result.extend(formatted_content)
            
    else:
        # 处理每个部分
        for i in range(1, len(parts), 2):
            if i >= len(parts):
                break
                
            header_text = parts[i]
            content_text = parts[i+1] if i+1 < len(parts) else ""
            
            # 添加格式化的标题，根据类型恢复"Key"前缀
            if "KEY_INCLUSION" in header_text:
                result.append("        Key Inclusion Criteria:")
            elif "KEY_EXCLUSION" in header_text:
                result.append("        Key Exclusion Criteria:")
            elif "INCLUSION" in header_text:
                result.append("        Inclusion Criteria:")
            else:
                result.append("        Exclusion Criteria:")
            
            result.append("")
            
            # 分析和格式化内容
            formatted_content = format_content_section(content_text, verbose)
            result.extend(formatted_content)
            
            if i < len(parts) - 2:
                result.append("")
                result.append("")
    
    combined = "\n".join(result)
    cleaned = re.sub(r'\n{3,}', '\n\n', combined)
    
    if verbose:
        print(f"输出文本 ({len(cleaned)} 字符):\n{cleaned[:100]}...")
    
    return cleaned

def format_content_section(content_text, verbose=False):
    """
    格式化内容部分
    - 分析和格式化分点条目
    - 处理导言和层级关系
    - 返回格式化后的行列表
    """
    # 预处理行，分析内容结构
    lines = content_text.strip().split('\n')
    if not lines:
        return []
        
    if verbose:
        print(f"行数: {len(lines)}")
        for i, line in enumerate(lines[:5]):
            print(f"Line {i}: {line[:50]}...")
    
    # 预分析行以检测分点符号和导言模式
    analyzed_lines = []
    
    # 首先收集所有行的基本信息
    for line in lines:
        if not line.strip(): continue  # 忽略空行
        leading_spaces = len(line) - len(line.lstrip(' '))
        clean_line = line.strip()
        
        # 识别分点符号类型 - 更宽松的匹配
        bullet_match = re.match(r'^(\d+\.\s|\*\s|-\s|[a-zA-Z]\.\s|\([a-zA-Z]\)\s)(.*)', clean_line)
        is_list_item = bool(bullet_match)
        
        # 检测是否为额外的大分类标题（不带分点符号但包含Inclusion/Exclusion Criteria字样）
        is_section_header = False
        if not is_list_item and re.search(r'(?i)(inclusion|exclusion)\s+criteria', clean_line):
            is_section_header = True
        
        if is_list_item:
            bullet_prefix = bullet_match.group(1)
            content_after_bullet = bullet_match.group(2)
            
            # 确定分点类型分组
            bullet_group = None
            if re.match(r'^\d+\.\s', bullet_prefix):
                bullet_group = "numeric"
            elif re.match(r'^[a-zA-Z]\.\s', bullet_prefix):
                bullet_group = "alphabetic"
            elif re.match(r'^\([a-zA-Z]\)\s', bullet_prefix):
                bullet_group = "alphabetic_paren"
            elif bullet_prefix == "* ":
                bullet_group = "star"
            elif bullet_prefix == "- ":
                bullet_group = "dash"
            
            analyzed_lines.append({
                'original': line,
                'content': clean_line,
                'content_after_bullet': content_after_bullet,
                'indent': leading_spaces,
                'is_list_item': True,
                'bullet_type': bullet_prefix,
                'bullet_group': bullet_group,
                'has_colon': clean_line.endswith(':'),
                'is_intro': False,  # 初始默认不是导言
                'is_section_header': False  # 分点项不是大分类标题
            })
        else:
            analyzed_lines.append({
                'original': line,
                'content': clean_line,
                'content_after_bullet': None,
                'indent': leading_spaces,
                'is_list_item': False,
                'bullet_type': None,
                'bullet_group': None,
                'has_colon': clean_line.endswith(':'),
                'is_intro': False,
                'is_section_header': is_section_header  # 是否为大分类标题
            })
    
    if not analyzed_lines:
        return []
    
    if verbose:
        print(f"分析行数: {len(analyzed_lines)}")
        print(f"分点行数: {sum(1 for line in analyzed_lines if line['is_list_item'])}")
        print(f"大分类标题数: {sum(1 for line in analyzed_lines if line['is_section_header'])}")

    # Count bullet groups to help identify introductory phrases correctly
    bullet_group_counts = {}
    for line in analyzed_lines:
        if line['bullet_group']:
            group = line['bullet_group']
            bullet_group_counts[group] = bullet_group_counts.get(group, 0) + 1
            
    # More robust intro phrase detection
    for line in analyzed_lines:
        if line['is_list_item'] and line['has_colon']:
            # Case 1: Explicit intro phrases like "Subjects must:", "Subjects may not:"
            # We check the content *after* the bullet to be more general.
            content_to_check = line['content_after_bullet'] if line['content_after_bullet'] else line['content']
            if re.search(r'(?i)^\s*(subjects\s+(must|may\s+not)|participants\s+must)\s*:', content_to_check):
                line['is_intro'] = True
                continue # Found, so skip to next line

            # Case 2: The "only item of its type" logic as a fallback for generic headers like "1. Title:"
            group = line['bullet_group']
            if bullet_group_counts.get(group, 0) == 1:
                line['is_intro'] = True
    
    # 进行层级分析和格式化
    formatted_lines = []
    level_stack = [] # Stack to track (indent, bullet_group)
    
    # 标记所有层级
    for idx, line in enumerate(analyzed_lines):
        # After an intro phrase, reset the leveling context to start a new top-level list
        if idx > 0 and analyzed_lines[idx-1]['is_intro']:
            level_stack.clear()

        # 如果是大分类标题，标记特殊级别
        if line['is_section_header']:
            level = -1
        elif line['is_list_item']:
            # 使用基于缩进的堆栈方法来确定层级
            current_indent = line['indent']
            current_bullet_group = line['bullet_group']

            # 如果堆栈为空或当前缩进大于栈顶缩进，则为新层级
            if not level_stack or current_indent > level_stack[-1][0]:
                level_stack.append((current_indent, current_bullet_group))
            else:
                # 如果缩进减少或不变，则弹出堆栈直到找到合适的父层级
                while level_stack and current_indent < level_stack[-1][0]:
                    level_stack.pop()
                
                # 如果缩进与栈顶相同，但分点类型不同，视为同级的新列表（弹出旧的）
                if level_stack and current_indent == level_stack[-1][0] and current_bullet_group != level_stack[-1][1]:
                    level_stack.pop()
                    level_stack.append((current_indent, current_bullet_group))
                elif not level_stack or current_indent > level_stack[-1][0] : # Popped back to a parent, this is a new sub-level
                     level_stack.append((current_indent, current_bullet_group))


            level = len(level_stack) - 1
        else:
            # 非分点行，根据上下文确定层级
            if idx > 0:
                if analyzed_lines[idx-1]['is_intro']:
                    level = 1  # 导言后的行是次级
                elif re.match(r'^\d+\s+Days$', line['content']) or line['content'].startswith('Note:'):
                    level = analyzed_lines[idx-1].get('level', 0) # Note/Days should be at same level as previous item
                else:
                    # 继承上一行的层级
                    level = analyzed_lines[idx-1].get('level', 0)
            else:
                level = 0
        
        line['level'] = level

    # 标记导言后的第一个条目，这些条目不应该在前面添加空行
    first_item_after_intro = set()
    for idx in range(1, len(analyzed_lines)):
        if analyzed_lines[idx-1]['is_intro']:
            first_item_after_intro.add(idx)
    
    # 格式化行
    for idx, line in enumerate(analyzed_lines):
        content = line['content']
        level = line['level']
        
        # 处理大分类标题
        if line['is_section_header']:
            # 大分类标题前添加空行（如果不是第一行）
            if idx > 0 and formatted_lines:
                formatted_lines.append("")
            
            # 添加大分类标题，使用与主标题相同的缩进（8个空格）
            formatted_lines.append("        " + content)
            
            # 大分类标题后添加空行
            formatted_lines.append("")
            continue
        
        # 正常处理其他行
        # 确定缩进
        base_indent = "          "  # 10个空格
        level_indent = "   "         # 每级增加3个空格
        indent = base_indent + level_indent * level
        
        # 特殊处理导言
        if line['is_intro']:
            # 导言移除分点符号
            formatted_content = content[len(line['bullet_type']):] if line['is_list_item'] else content
            formatted_lines.append(indent + formatted_content)
        else:
            # 非导言保持原样
            formatted_lines.append(indent + content)
        
        # 在顶级分点之间添加空行
        if idx < len(analyzed_lines) - 1:
            next_line = analyzed_lines[idx + 1]
            next_level = next_line['level']
            next_is_list_item = next_line['is_list_item']
            next_is_section_header = next_line['is_section_header']
            
            # 不需要在大分类标题前添加空行，因为大分类标题的处理逻辑中已经添加了空行
            if not next_is_section_header:
                # 只要下一个是顶级分点，就空一行（除非是紧跟导言的第一项）
                if next_level == 0 and next_is_list_item and (idx + 1) not in first_item_after_intro:
                    formatted_lines.append("")
    
    return formatted_lines

def process_csv_file(input_path, output_path, verbose=False):
    """
    读取CSV文件，应用格式化到criteria列，并写入新文件
    """
    try:
        # 增加CSV字段大小限制，以处理大的文本字段
        # 这是一个处理CSV文件中大文本字段时的常见问题
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
            clean_header = [col.strip('"\'') for col in header]
            writer.writerow(clean_header)

            criteria_index = -1
            possible_criteria_columns = ['criteria', 'eligibility_criteria', 'eligibilityCriteria']
            
            for col_name in possible_criteria_columns:
                try:
                    criteria_index = clean_header.index(col_name)
                    print(f"Found criteria column as '{col_name}' at index {criteria_index}")
                    break
                except ValueError:
                    continue
            
            if criteria_index == -1:
                print(f"Error: No criteria column found in {input_path}")
                print(f"Available columns: {clean_header}")
                for row in reader:
                    writer.writerow(row)
                return

            # 处理每一行
            row_count = 0
            for row in reader:
                row_count += 1
                if len(row) > criteria_index:
                    original_criteria = row[criteria_index]
                    if verbose:
                        print(f"处理第 {row_count} 行, 原始长度: {len(original_criteria)}")
                    
                    reformatted_criteria = normalize_criteria_structure(original_criteria, verbose)
                    
                    if verbose:
                        print(f"格式化后长度: {len(reformatted_criteria)}")
                        if len(reformatted_criteria) < 100:
                            print(f"警告: 可能丢失内容! 格式化后文本: {reformatted_criteria}")
                    
                    row[criteria_index] = reformatted_criteria
                writer.writerow(row)

        print(f"Successfully reformatted '{input_path}' and saved to '{output_path}'")
        print(f"Processed {row_count} rows")

    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'")
    except Exception as e:
        print(f"An unexpected error occurred while processing '{input_path}': {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    verbose = '--verbose' in sys.argv
    
    files_to_process = [
        'SFT_train_set/SFT_train_set.csv'
    ]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    for file_path in files_to_process:
        abs_input_path = os.path.join(project_root, file_path)
        
        base, ext = os.path.splitext(abs_input_path)
        # Use a new suffix for the final version
        abs_output_path = f"{base}_reformat{ext}"
        
        process_csv_file(abs_input_path, abs_output_path, verbose) 