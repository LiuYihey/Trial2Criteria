import csv
import os

def remove_bom(input_file, output_file=None):
    """移除CSV文件的BOM标记"""
    if output_file is None:
        # 如果没有指定输出文件，使用临时文件并替换原文件
        output_file = input_file + '.nobom'
        replace_original = True
    else:
        replace_original = False
    
    # 读取带有BOM的文件
    with open(input_file, 'r', encoding='utf-8-sig') as infile:
        # 'utf-8-sig'编码会自动跳过BOM
        reader = csv.reader(infile)
        header = next(reader)
        rows = list(reader)
    
    # 写入不带BOM的文件
    with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)
        writer.writerows(rows)
    
    if replace_original:
        # 备份原文件
        backup_file = input_file + '.bak'
        if os.path.exists(backup_file):
            os.remove(backup_file)
        os.rename(input_file, backup_file)
        os.rename(output_file, input_file)
        print(f"已移除BOM标记并替换原文件。原文件备份为: {backup_file}")
    else:
        print(f"已移除BOM标记并保存到: {output_file}")

if __name__ == "__main__":
    # 指定要处理的CSV文件路径
    input_file = 'Retrieval_Base\\Original\\RAG_base_phase2_raw.csv'
    remove_bom(input_file) 