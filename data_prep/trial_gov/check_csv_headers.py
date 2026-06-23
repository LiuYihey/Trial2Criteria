import csv

input_file = 'Retrieval_Base\\Original\\RAG_base_phase2_raw.csv'

with open(input_file, 'r', encoding='latin1') as infile:
    reader = csv.reader(infile)
    headers = next(reader)  # 读取第一行作为表头
    print("CSV表头列名:")
    for i, header in enumerate(headers):
        print(f"{i}: '{header}'")
    
    # 读取第一行数据
    try:
        first_row = next(reader)
        print("\n第一行数据:")
        for i, value in enumerate(first_row):
            print(f"{i}: '{value}'")
    except StopIteration:
        print("CSV文件中没有数据行") 