import pandas as pd
import os

# 定义数据文件的路径
DRUG_FEATURES_PATH = "drug_features.csv"

def load_data(file_path):
    """加载CSV数据文件"""
    if not os.path.exists(file_path):
        print(f"错误：数据文件 '{file_path}' 不存在。")
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        print(f"成功加载文件: {file_path}")
        return df
    except Exception as e:
        print(f"加载 '{file_path}' 文件时出错: {e}")
        return None

def find_drug_features(df, drug_name):
    """
    在DataFrame中查找并打印药物的特征信息。
    匹配是精确的，但忽略大小写。
    """
    # 创建一个用于不区分大小写比较的临时列
    df_temp = df.copy()
    df_temp['name_lower'] = df_temp['node_name'].str.lower()
    
    # 查找匹配的行
    result_row = df_temp[df_temp['name_lower'] == drug_name.lower()]
    
    if result_row.empty:
        print(f"\n未找到名为 '{drug_name}' 的药物。")
        return

    # 提取第一条匹配记录
    features = result_row.iloc[0].to_dict()
    
    print(f"\n--- '{features.get('node_name')}' 的特征信息 ---")
    for key, value in features.items():
        # 忽略用于搜索的临时列
        if key == 'name_lower':
            continue
        # 检查值是否有效且不为空
        if pd.notna(value) and str(value).strip() != '':
            print(f"[{key}]:")
            print(f"  {value}\n")
    print("------------------------------------------")

def main():
    """主函数，用于命令行交互"""
    drug_df = load_data(DRUG_FEATURES_PATH)
    if drug_df is None:
        return
        
    try:
        while True:
            drug_name = input("\n请输入药物名称 (输入 'q' 退出): ")
            if drug_name.lower() == 'q':
                break
            find_drug_features(drug_df, drug_name)
    except KeyboardInterrupt:
        print("\n程序已退出。")
    except Exception as e:
        print(f"发生未知错误: {e}")

if __name__ == "__main__":
    main() 