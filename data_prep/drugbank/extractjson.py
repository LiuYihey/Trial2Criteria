def split_lines(input_path, output_path, start_line, end_line):
    """
    将 input_path 文件的 start_line 到 end_line 行写入 output_path。
    
    参数:
      input_path (str): 原始大文件路径
      output_path (str): 输出文件路径
      start_line (int): 起始行数
      end_line (int): 结束行数
    """
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        for i, line in enumerate(infile):
            if i >= start_line and i <= end_line:
                outfile.write(line)
            if i > end_line:
                break

if __name__ == "__main__":
    split_lines(
        input_path=r'C:\Users\86137\Desktop\Trial数据\self\drugbank\output.json',
        output_path=r'C:\Users\86137\Desktop\Trial数据\self\drugbank\sample_lines.json',
        start_line = 90000,
        end_line = 110000
    )
    print("已保存到 sample_lines.json")
