import pandas as pd
import re
import os

def filter_nct_in_criteria(input_path, output_path):
    """
    Reads a CSV file, removes rows where 'eligibilityCriteria' contains another NCT number,
    and saves the result to a new CSV file.

    Args:
        input_path (str): The path to the input CSV file.
        output_path (str): The path to the output CSV file.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(input_path)

        # Check if 'eligibilityCriteria' column exists
        if 'eligibilityCriteria' not in df.columns:
            print(f"错误: 在 {input_path} 中未找到 'eligibilityCriteria' 列")
            return

        # Define the regex pattern to find NCT numbers
        nct_pattern = re.compile(r'NCT\d+')

        # Create a boolean mask for rows to keep.
        # We want to keep rows that DO NOT contain the pattern, so we use the '~' operator.
        # `na=False` ensures that NaN values in the column are kept.
        mask = ~df['eligibilityCriteria'].str.contains(nct_pattern, na=False)

        # 打印被滤除的样本ID
        print(df[~mask])

        # Apply the mask to filter the DataFrame
        df_filtered = df[mask]

        # Save the filtered DataFrame to a new CSV file, ensuring BOM is handled for Excel compatibility
        df_filtered.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"成功过滤文件并保存至 {output_path}")
        print(f"原始行数: {len(df)}")
        print(f"过滤后行数: {len(df_filtered)}")
        print(f"移除的行数: {len(df) - len(df_filtered)}")

    except FileNotFoundError:
        print(f"错误: 文件 {input_path} 未找到。")
    except Exception as e:
        print(f"发生意外错误: {e}")

if __name__ == "__main__":
    # Define file paths
    input_file = 'SFT_train_set/SFT_train_set_831.csv'
    output_file = 'SFT_train_set/SFT_train_set_rmNCT.csv'

    # Execute the function
    filter_nct_in_criteria(input_file, output_file) 