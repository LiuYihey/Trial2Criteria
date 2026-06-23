import requests
from urllib.parse import quote
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import pandas as pd
import time
import os

# 1. 准备一个带重试和长超时的 Session
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["HEAD", "GET", "OPTIONS"]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_cids(name):
    """小分子：名称→CID 列表，如果 404 则返回空列表。"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(name)}/cids/JSON"
    try:
        resp = session.get(url, timeout=15)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            return []
        raise
    data = resp.json()
    return data.get("IdentifierList", {}).get("CID", [])

def get_compound_synonyms(cid):
    """小分子：CID→同义词列表"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/synonyms/JSON"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    info = resp.json().get("InformationList", {}).get("Information", [])
    return info[0].get("Synonym", []) if info else []

def get_sids(name):
    """大分子：名称→SID 列表"""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/name/{quote(name)}/sids/JSON"
    resp = session.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    # 优先 IdentifierList.SID
    sids = data.get("IdentifierList", {}).get("SID")
    if sids:
        return sids
    # 回退到 InformationList
    info = data.get("InformationList", {}).get("Information", [])
    return info[0].get("SID", []) if info else []

def get_substance_synonyms(sid):
    """
    大分子：尝试 PUG REST 同义词接口；如果超时或响应过长，
    切到 PUGView 分页接口只取 Names and Identifiers → Synonyms。
    """
    # 先尝试简单的 synonyms/JSON
    url1 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/substance/sid/{sid}/synonyms/JSON"
    try:
        resp = session.get(url1, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # 短列表优先从 IdentifierList
        synonyms = data.get("IdentifierList", {}).get("Synonym")
        if synonyms:
            return synonyms
        # 回退到 InformationList
        info = data.get("InformationList", {}).get("Information", [])
        return info[0].get("Synonym", []) if info else []
    except (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
        # 超时的话切换到 PUGView
        url2 = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug_view/data/substance/sid/{sid}/JSON"
        resp2 = session.get(url2, timeout=30)
        resp2.raise_for_status()
        sections = resp2.json().get("Record", {}).get("Section", [])
        # 在 Section 里找 Names and Identifiers → Synonyms
        for sec in sections:
            if sec.get("TOCHeading") == "Names and Identifiers":
                for sub in sec.get("Section", []):
                    if sub.get("TOCHeading") == "Synonyms":
                        # 每个 Information 里一堆字符串
                        return [item["Value"]["StringWithMarkup"][0]["String"]
                                for item in sub.get("Information", [])]
        return []

def get_all_synonyms(name):
    """
    先尝试小分子 CID 路径，不行就走大分子 SID 路径。
    最终去重、排序后返回同义词列表。
    """
    # 1. CID 路径
    cids = get_cids(name)
    if cids:
        syns = []
        for cid in cids:
            syns += get_compound_synonyms(cid)
        if syns:
            return sorted(set(syns))

    # 2. SID 路径
    sids = get_sids(name)
    if sids:
        syns = []
        for sid in sids:
            syns += get_substance_synonyms(sid)
        if syns:
            return sorted(set(syns))

    # 全部空
    raise RuntimeError(f"No synonyms found for {name!r} via either CID or SID paths.")

def process_drug_list(input_csv, output_csv):
    """
    处理药物列表并保存同义词，实时保存结果
    
    Args:
        input_csv: 输入CSV文件路径，第一列为药物名
        output_csv: 输出CSV文件路径
    """
    # 读取输入CSV文件
    df = pd.read_csv(input_csv)
    drug_column = df.columns[0]  # 获取第一列的列名
    
    # 创建输出DataFrame
    output_df = pd.DataFrame(columns=['Drug_Name', 'Synonyms'])
    
    # 如果输出文件已存在，读取已有结果
    if os.path.exists(output_csv):
        try:
            output_df = pd.read_csv(output_csv)
            print(f"已读取现有结果文件，继续处理剩余药物...")
        except Exception as e:
            print(f"读取现有结果文件失败: {str(e)}")
    
    # 获取已处理的药物列表
    processed_drugs = set(output_df['Drug_Name'].tolist())
    
    # 处理每个药物
    for drug in df[drug_column]:
        # 跳过已处理的药物
        if drug in processed_drugs:
            print(f"跳过已处理的药物: {drug}")
            continue
            
        try:
            print(f"正在处理药物: {drug}")
            synonyms = get_all_synonyms(drug)
            # 将同义词列表转换为字符串，用逗号分隔
            synonyms_str = ", ".join(synonyms)
            
            # 添加新结果
            new_row = pd.DataFrame({
                'Drug_Name': [drug],
                'Synonyms': [synonyms_str]
            })
            output_df = pd.concat([output_df, new_row], ignore_index=True)
            
            # 实时保存到CSV文件
            output_df.to_csv(output_csv, index=False, encoding='utf-8')
            
            # 添加延时以避免请求过快
            # time.sleep(1)
            
        except Exception as e:
            print(f"处理药物 {drug} 时出错: {str(e)}")
            # 即使出错也保存空结果
            new_row = pd.DataFrame({
                'Drug_Name': [drug],
                'Synonyms': [""]
            })
            output_df = pd.concat([output_df, new_row], ignore_index=True)
            output_df.to_csv(output_csv, index=False, encoding='utf-8')
            print(f"已保存药物 {drug} 的错误结果")
    
    print(f"所有药物处理完成，结果已保存到: {output_csv}")

if __name__ == "__main__":
    # 设置输入输出文件路径
    input_file = "drugbank_data_v1.csv"  # 请确保此文件存在
    output_file = "drug_synonyms.csv"
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 {input_file} 不存在")
    else:
        process_drug_list(input_file, output_file)
