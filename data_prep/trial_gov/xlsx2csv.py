import pandas as pd

def excel_to_csv(excel_file, sheet_name, csv_file):
    # 读取 Excel 文件中的特定工作表
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    # 保存为 CSV 文件，并指定编码格式为 UTF-8
    df.to_csv(csv_file, index=False, encoding='utf-8')

# 示例用法
if __name__ == "__main__":
    excel_to_csv(r'C:\Users\86137\Desktop\Trial_project\Pubmed\drug\drugbank_data_v1.xlsx', 'drugbank_data_v1', r'C:\Users\86137\Desktop\Trial_project\Pubmed\drug\drugbank_data_v1.csv')
