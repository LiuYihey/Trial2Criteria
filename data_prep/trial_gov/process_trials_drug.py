
# -*- coding: utf-8 -*-

import os
from extractdrug import process_csv_interventions

if __name__ == "__main__":
    # 输入和输出文件路径
    input_csv = r"C:\Users\86137\Desktop\Trial数据\self\trial_gov\Retrieval_Base\Original\RAG_base_phase1_raw.csv"
    output_csv = r"C:\Users\86137\Desktop\Trial数据\self\trial_gov\Retrieval_Base\Original\extracted_drugs_phase1.csv"
    
    # 处理CSV文件
    process_csv_interventions(input_csv, output_csv) 