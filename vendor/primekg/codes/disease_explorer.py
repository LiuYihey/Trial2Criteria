import os
import pandas as pd
from collections import defaultdict

# 定义数据文件的本地路径
KG_FILE_PATH = "kg.csv"

def find_related_info(disease_name: str, kg_df: pd.DataFrame, allowed_relations: list = None, allowed_sources: list = None):
    """
    在知识图谱中查找与指定疾病相关的信息。

    :param disease_name: 要查询的疾病名称 (字符串)。
    :param kg_df: 包含知识图谱数据的Pandas DataFrame。
    :param allowed_relations: (可选) 一个关系类型字符串的列表，用于筛选结果。如果为 None，则不按关系筛选。
    :param allowed_sources: (可选) 一个数据源字符串的列表，用于筛选结果。如果为 None，则不按来源筛选。
    :return: 一个字典，键是实体类型，值是与该疾病相关的实体名称列表。
    """
    print(f"\n正在搜索与 '{disease_name}' 相关的信息...")
    # 使用 .str.contains() 进行不区分大小写的搜索
    disease_mask = (kg_df['x_name'].str.contains(disease_name, case=False, na=False)) | \
                   (kg_df['y_name'].str.contains(disease_name, case=False, na=False))
    
    related_edges = kg_df[disease_mask]

    # 根据提供的关系类型列表进行筛选
    if allowed_relations:
        related_edges = related_edges[related_edges['relation'].isin(allowed_relations)]

    # 根据提供的数据源列表进行筛选
    if allowed_sources:
        source_mask = (related_edges['x_source'].isin(allowed_sources)) | \
                      (related_edges['y_source'].isin(allowed_sources))
        related_edges = related_edges[source_mask]

    if related_edges.empty:
        print("未找到相关信息。")
        return {}

    related_entities = defaultdict(set)
    
    # 确定输入的疾病节点的准确名称和类型，以避免将其自身包含在结果中
    # 我们假设最常见的名称是正确的那个
    try:
        if kg_df[kg_df['x_name'].str.contains(disease_name, case=False, na=False)].shape[0] > 0:
            exact_disease_name = kg_df[kg_df['x_name'].str.contains(disease_name, case=False, na=False)]['x_name'].value_counts().idxmax()
            disease_type = kg_df[kg_df['x_name'] == exact_disease_name]['x_type'].iloc[0]
        else:
            exact_disease_name = kg_df[kg_df['y_name'].str.contains(disease_name, case=False, na=False)]['y_name'].value_counts().idxmax()
            disease_type = kg_df[kg_df['y_name'] == exact_disease_name]['y_type'].iloc[0]
    except (IndexError, KeyError):
        # 如果找不到准确的疾病名称或类型，则使用一个不会匹配任何内容的占位符
        exact_disease_name, disease_type = "___PLACEHOLDER___", "disease"


    for _, row in related_edges.iterrows():
        # 检查源节点 (x) 是否是目标疾病
        if str(row['x_name']).lower() == exact_disease_name.lower() and row['x_type'] == disease_type:
            # 如果是，则将目标节点 (y) 添加到结果中
            entity_type = row['y_type']
            entity_name = row['y_name']
            # 我们不希望结果中包含 "disease" 类型的实体
            if entity_type != 'disease':
                 related_entities[entity_type].add(entity_name)

        # 检查目标节点 (y) 是否是目标疾病
        if str(row['y_name']).lower() == exact_disease_name.lower() and row['y_type'] == disease_type:
            # 如果是，则将源节点 (x) 添加到结果中
            entity_type = row['x_type']
            entity_name = row['x_name']
            if entity_type != 'disease':
                related_entities[entity_type].add(entity_name)

    # 将集合转换为列表以便于输出
    return {k: list(v) for k, v in related_entities.items()}

if __name__ == "__main__":
    # 1. 检查数据文件是否存在
    if not os.path.exists(KG_FILE_PATH):
        print(f"错误：数据文件 '{KG_FILE_PATH}' 不存在。")
        print("请手动下载文件：")
        print("1. 在您的浏览器中打开以下链接:")
        print("   https://dataverse.harvard.edu/api/access/datafile/6180620")
        print(f"2. 将下载的文件保存到当前目录下，并命名为 '{KG_FILE_PATH}'。")
        print("3. 完成后，请重新运行此脚本。")
        exit()

    # 2. 加载知识图谱数据
    print("正在加载知识图谱文件 kg.csv... 这可能需要一些时间。")
    try:
        # 指定数据类型以减少内存使用
        dtype_spec = {
            'x_id': 'string', 'x_type': 'category', 'x_name': 'string', 'x_source': 'category',
            'relation': 'category', 'y_id': 'string', 'y_type': 'category', 'y_name': 'string',
            'y_source': 'category', 'display_relation': 'category', 'pmid': 'string'
        }
        kg_main_df = pd.read_csv(KG_FILE_PATH, low_memory=False, dtype=dtype_spec)
        print("知识图谱加载完成。")

    except (FileNotFoundError, pd.errors.ParserError) as e:
        print(f"加载 kg.csv 文件时出错: {e}")
        exit()

    # 3. 设置要查询的疾病名称
    # disease_to_search = "Parkinson's disease"
    # disease_to_search = "Alzheimer's disease"
    disease_to_search = "glioblastoma" # 您可以更改为任何您想查询的疾病

    # 4. (可选) 定义要筛选的关系和数据源
    # 如果列表为 None 或为空，则不进行筛选。
    # 示例关系: ['contraindicates_disease', 'treats_disease', 'presents_disease', 'causes_disease']
    # 示例来源: ['DrugBank', 'TTD', 'KEGG']
    target_relations = None
    target_sources = None

    # 5. 执行查询
    related_info = find_related_info(
        disease_to_search,
        kg_main_df,
        allowed_relations=target_relations,
        allowed_sources=target_sources
    )

    # 6. 格式化并打印结果
    if related_info:
        print(f"\n--- 与 '{disease_to_search}' 相关的实体信息 ---")
        # 添加筛选条件到标题
        if target_relations:
            print(f"筛选关系: {', '.join(target_relations)}")
        if target_sources:
            print(f"筛选来源: {', '.join(target_sources)}")
        
        for entity_type, entities in related_info.items():
            print(f"\n[{entity_type.upper()}] - 共 {len(entities)} 个")
            # 为了简洁，只显示前10个
            for entity in sorted(entities)[:10]:
                print(f"  - {entity}")
            if len(entities) > 10:
                print(f"  ... (还有 {len(entities) - 10} 个)")
        print("\n------------------------------------")
    else:
        print(f"数据库中没有找到与 '{disease_to_search}' 相关的明确信息。请检查疾病名称是否准确。") 