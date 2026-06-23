import csv
import json
import os

def get_successful_nct_ids(report_path):
    """
    Reads the report CSV and returns a set of nctIds where all columns (except the first) are '1'.
    """
    successful_ids = set()
    try:
        with open(report_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)  # Skip header
            for row in reader:
                if row and all(val.strip() == '1' for val in row[1:]):
                    successful_ids.add(row[0].strip())
    except FileNotFoundError:
        print(f"错误：报告文件未找到 at {report_path}")
    except Exception as e:
        print(f"读取报告文件时发生错误: {e}")
    return successful_ids

def filter_csv_file(input_path, output_path, valid_ids):
    """
    Filters a CSV file based on a set of valid nctIds.
    """
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            
            header = next(reader)
            writer.writerow(header)
            
            # Assuming nctId is the first column
            nct_id_index = 0
            
            filtered_count = 0
            for row in reader:
                if row and len(row) > nct_id_index and row[nct_id_index].strip() in valid_ids:
                    writer.writerow(row)
                    filtered_count += 1
            print(f"成功筛选并写入 {filtered_count} 条记录到 {output_path}")
    except FileNotFoundError:
        print(f"错误：输入CSV文件未找到 at {input_path}")
    except Exception as e:
        print(f"处理CSV文件时发生错误: {e}")

def filter_jsonl_file(input_path, output_path, valid_ids):
    """
    Filters a JSONL file based on a set of valid nctIds.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as infile, \
             open(output_path, 'w', encoding='utf-8') as outfile:
            filtered_count = 0
            for line in infile:
                try:
                    data = json.loads(line)
                    if data.get('nctId') in valid_ids:
                        outfile.write(line)
                        filtered_count += 1
                except json.JSONDecodeError:
                    print(f"警告：跳过无法解析的JSON行: {line.strip()}")
            print(f"成功筛选并写入 {filtered_count} 条记录到 {output_path}")
    except FileNotFoundError:
        print(f"错误：输入JSONL文件未找到 at {input_path}")
    except Exception as e:
        print(f"处理JSONL文件时发生错误: {e}")

def main():
    """
    Main function to orchestrate the filtering process.
    """
    # Define file paths
    report_file = 'RAG_context_batch_generation/test_trials_retrieval_report.csv'
    jsonl_input_file = 'RAG_context_batch_generation/test_trials_RAG_831.jsonl'
    csv_input_file = 'self/trial_gov/After_date_test_set/test_trials_831.csv'
    jsonl_output_file = 'RAG_context_batch_generation/test_trials_RAG_831_filtered.jsonl'
    csv_output_file = 'self/trial_gov/After_date_test_set/test_trials_831_filtered.csv'
    
    print("步骤 1: 从报告文件中提取所有列值均为1的样本号...")
    successful_nct_ids = get_successful_nct_ids(report_file)
    print(f"找到 {len(successful_nct_ids)} 个符合条件的样本号。")
    
    if not successful_nct_ids:
        print("未找到符合条件的样本号，处理结束。")
        return
        
    print("\n步骤 2: 筛选CSV文件...")
    filter_csv_file(csv_input_file, csv_output_file, successful_nct_ids)
    
    print("\n步骤 3: 筛选JSONL文件...")
    filter_jsonl_file(jsonl_input_file, jsonl_output_file, successful_nct_ids)
    
    print("\n所有处理已完成。")

if __name__ == "__main__":
    main() 