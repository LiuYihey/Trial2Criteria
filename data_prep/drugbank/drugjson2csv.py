import json
import csv
import requests
import os
from collections import defaultdict
import re

CSV_COLUMNS = [
    'name', 'products_name', 'SMILES', 'molecular_formula', 'description', 'state',
    'indication', 'pharmacodynamics', 'mechanism-of-action', 'toxicity', 'metabolism',
    'absorption', 'half-life', 'protein-binding', 'route-of-elimination',
    'volume-of-distribution', 'clearance', 'food-interaction', 'pubmed-id'
]
NAMESPACE = "{http://www.drugbank.ca}"

def clean_value(value):
    """清洗值中的命名空间和#text"""
    if value is None:
        return 'none'
    if isinstance(value, dict) and '#text' in value:
        text = str(value['#text']).replace(NAMESPACE, '')
        # 删除[...]格式的引用标记
        text = re.sub(r'\[[^\]]*\]', '', text)
        return text
    elif isinstance(value, str):
        text = value.replace(NAMESPACE, '')
        # 删除[...]格式的引用标记
        text = re.sub(r'\[[^\]]*\]', '', text)
        return text
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        # 处理列表类型的值
        cleaned_values = []
        for item in value:
            if item is not None:
                cleaned_values.append(clean_value(item))
        return ', '.join(filter(None, cleaned_values))
    return str(value).replace(NAMESPACE, '')

def get_pubchem_data(name):
    """从PubChem获取化合物信息"""
    if not name or name == 'none':
        return 'none', 'none'
    
    try:
        # 获取CID
        cid_response = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON",
            timeout=10
        )
        if cid_response.status_code != 200:
            return 'none', 'none'
        
        cid_data = cid_response.json()
        if 'IdentifierList' not in cid_data or 'CID' not in cid_data['IdentifierList'] or not cid_data['IdentifierList']['CID']:
            return 'none', 'none'
            
        cid = cid_data['IdentifierList']['CID'][0]
        
        # 获取属性
        prop_response = requests.get(
            f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/SMILES,MolecularFormula/JSON",
            timeout=10
        )
        if prop_response.status_code != 200:
            return 'none', 'none'
            
        props = prop_response.json()['PropertyTable']['Properties'][0]
        return props.get('SMILES', 'none'), props.get('MolecularFormula', 'none')
    except Exception as e:
        print(f"PubChem API错误（{name}）：{str(e)}")
        return 'none', 'none'

def find_drug_elements(json_data):
    """深度搜索找到所有drug元素"""
    drugs = []
    
    def search_drug(data, path=""):
        if isinstance(data, dict):
            for key, value in data.items():
                # 找到drug节点
                if key == f"{NAMESPACE}drug":
                    if isinstance(value, list):
                        drugs.extend(value)
                    else:
                        drugs.append(value)
                else:
                    search_drug(value, f"{path}.{key}")
        elif isinstance(data, list):
            for item in data:
                search_drug(item, path)
    
    search_drug(json_data)
    return drugs

def process_drug(drug_data):
    """处理单个drug数据"""
    result = {col: 'none' for col in CSV_COLUMNS}
    
    # 从例子来看，name位于drugbank-id后面
    def find_name_after_drugbank_id(data):
        if not isinstance(data, dict):
            return None
        # 查找"{http://www.drugbank.ca}drugbank-id"是否存在
        if f"{NAMESPACE}drugbank-id" in data:
            # 并且"{http://www.drugbank.ca}name"也存在，且都在同一层级
            if f"{NAMESPACE}name" in data:
                name_value = data[f"{NAMESPACE}name"]
                return clean_value(name_value)
        return None
    
    # 获取正确的description（位于name字段后面的description）
    def find_correct_description(data):
        if not isinstance(data, dict):
            return None
        # 检查是否有name字段
        if f"{NAMESPACE}name" in data:
            # 检查name后面是否有description字段
            if f"{NAMESPACE}description" in data:
                desc_value = data[f"{NAMESPACE}description"]
                return clean_value(desc_value)
        return None
    
    # 查找products下的所有name
    def find_product_names(data):
        product_names = set()
        
        def recursive_product_search(node, in_products=False, in_product=False):
            if isinstance(node, dict):
                # 检查是否在products内
                new_in_products = in_products
                new_in_product = in_product
                
                if f"{NAMESPACE}products" in node:
                    new_in_products = True
                    products_data = node[f"{NAMESPACE}products"]
                    
                    # 添加空值检查
                    if products_data is not None:
                        # 处理product或product列表
                        if f"{NAMESPACE}product" in products_data:
                            product_data = products_data[f"{NAMESPACE}product"]
                            
                            # 如果是列表
                            if isinstance(product_data, list):
                                for product in product_data:
                                    if isinstance(product, dict) and f"{NAMESPACE}name" in product:
                                        name = clean_value(product[f"{NAMESPACE}name"])
                                        if name:
                                            product_names.add(name)
                            # 如果是单个产品
                            elif isinstance(product_data, dict) and f"{NAMESPACE}name" in product_data:
                                name = clean_value(product_data[f"{NAMESPACE}name"])
                                if name:
                                    product_names.add(name)
                
                # 递归处理所有键
                for key, value in node.items():
                    recursive_product_search(value, new_in_products, new_in_product)
            
            # 如果是列表，递归处理每个元素
            elif isinstance(node, list):
                for item in node:
                    recursive_product_search(item, in_products, in_product)
        
        recursive_product_search(data)
        return product_names
    
    # 查找general-references下的所有pubmed-id
    def find_pubmed_ids(data):
        pubmed_ids = set()
        
        def recursive_reference_search(node):
            if isinstance(node, dict):
                # 直接检查是否有general-references字段
                if f"{NAMESPACE}general-references" in node:
                    ref_data = node[f"{NAMESPACE}general-references"]
                    
                    # 添加空值检查
                    if ref_data is not None:
                        # 检查是否直接包含pubmed-id
                        if isinstance(ref_data, dict):
                            # 在articles下查找article
                            if f"{NAMESPACE}articles" in ref_data:
                                articles_data = ref_data[f"{NAMESPACE}articles"]
                                
                                # 添加空值检查
                                if articles_data is not None and isinstance(articles_data, dict):
                                    # 处理article列表
                                    if f"{NAMESPACE}article" in articles_data:
                                        articles = articles_data[f"{NAMESPACE}article"]
                                        if isinstance(articles, list):
                                            for article in articles:
                                                if isinstance(article, dict) and f"{NAMESPACE}pubmed-id" in article:
                                                    pubmed_id = clean_value(article[f"{NAMESPACE}pubmed-id"])
                                                    if pubmed_id:
                                                        pubmed_ids.add(pubmed_id)
                                        elif isinstance(articles, dict) and f"{NAMESPACE}pubmed-id" in articles:
                                            pubmed_id = clean_value(articles[f"{NAMESPACE}pubmed-id"])
                                            if pubmed_id:
                                                pubmed_ids.add(pubmed_id)
                            
                            # 在reference-list下查找articles
                            elif f"{NAMESPACE}reference-list" in ref_data:
                                ref_list = ref_data[f"{NAMESPACE}reference-list"]
                                
                                # 添加空值检查
                                if ref_list is not None and isinstance(ref_list, dict):
                                    # 处理reference列表
                                    if f"{NAMESPACE}reference" in ref_list:
                                        refs = ref_list[f"{NAMESPACE}reference"]
                                        if isinstance(refs, list):
                                            for ref in refs:
                                                if isinstance(ref, dict) and f"{NAMESPACE}pubmed-id" in ref:
                                                    pubmed_id = clean_value(ref[f"{NAMESPACE}pubmed-id"])
                                                    if pubmed_id:
                                                        pubmed_ids.add(pubmed_id)
                                        elif isinstance(refs, dict) and f"{NAMESPACE}pubmed-id" in refs:
                                            pubmed_id = clean_value(refs[f"{NAMESPACE}pubmed-id"])
                                            if pubmed_id:
                                                pubmed_ids.add(pubmed_id)
                            
                            # 直接检查general-references下的pubmed-id
                            elif f"{NAMESPACE}pubmed-id" in ref_data:
                                pubmed_id = clean_value(ref_data[f"{NAMESPACE}pubmed-id"])
                                if pubmed_id:
                                    pubmed_ids.add(pubmed_id)
                
                # 递归处理所有键
                for key, value in node.items():
                    # 不要深入搜索，只在第一层检查general-references
                    # 避免匹配其他父级（如target）下的pubmed-id
                    if isinstance(value, dict) or isinstance(value, list):
                        recursive_reference_search(value)
            
            # 如果是列表，递归处理每个元素
            elif isinstance(node, list):
                for item in node:
                    recursive_reference_search(item)
        
        recursive_reference_search(data)
        return pubmed_ids
    
    # 提取普通字段
    def extract_regular_fields(data):
        if not isinstance(data, dict):
            return
            
        for col in CSV_COLUMNS:
            if col not in ['name', 'products_name', 'pubmed-id', 'SMILES', 'molecular_formula', 'description']:
                namespace_col = f"{NAMESPACE}{col}"
                if namespace_col in data:
                    value = data[namespace_col]
                    result[col] = clean_value(value)
        
        # 递归处理嵌套结构
        for key, value in data.items():
            if isinstance(value, dict):
                extract_regular_fields(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        extract_regular_fields(item)
    
    # 1. 查找name
    name = find_name_after_drugbank_id(drug_data)
    if name:
        result['name'] = name
    
    # 2. 查找product_names
    product_names = find_product_names(drug_data)
    if product_names:
        result['products_name'] = ', '.join(product_names)
    
    # 3. 查找pubmed_ids
    pubmed_ids = find_pubmed_ids(drug_data)
    if pubmed_ids:
        result['pubmed-id'] = ', '.join(pubmed_ids)
    
    # 4. 获取正确的description
    description = find_correct_description(drug_data)
    if description:
        result['description'] = description
    
    # 5. 提取其他普通字段
    extract_regular_fields(drug_data)
    
    # 6. 获取PubChem数据
    if result['name'] != 'none':
        smiles, formula = get_pubchem_data(result['name'])
        result['SMILES'] = smiles
        result['molecular_formula'] = formula
    
    return result

def json_to_csv(input_file, output_file, max_drugs=None):
    """将JSON文件转换为CSV文件
    
    Args:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出CSV文件路径
        max_drugs (int, optional): 最大处理的drug数量，None表示处理所有drug
    """
    # 验证输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误：输入文件 {input_file} 不存在")
        return
    
    # 验证文件大小
    file_size = os.path.getsize(input_file)
    if file_size == 0:
        print("错误：输入文件为空")
        return
    
    try:
        print(f"开始解析文件：{input_file}")
        # 读取文件内容并保存行号信息
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            json_content = ''.join(lines)
            try:
                json_data = json.loads(json_content)
                print("JSON文件加载成功")
            except json.JSONDecodeError as e:
                print(f"JSON解析错误：{str(e)}")
                print(f"错误位置：第{e.lineno}行，第{e.colno}列")
                return
        
        # 找到所有drug元素
        drugs = find_drug_elements(json_data)
        print(f"找到 {len(drugs)} 个drug元素")
        
        if not drugs:
            print("警告：未找到drug元素！")
            return
        
        # 如果设置了max_drugs，只处理指定数量的drug
        if max_drugs is not None:
            drugs = drugs[:max_drugs]
            print(f"将处理前 {len(drugs)} 个drug元素")
        
        # 处理每个drug元素
        results = []
        for i, drug in enumerate(drugs):
            try:
                print(f"处理drug元素 {i+1}/{len(drugs)}")
                result = process_drug(drug)
                results.append(result)
            except Exception as e:
                # 获取错误发生时的drug信息
                drug_info = json.dumps(drug, indent=2)
                print(f"\n处理第 {i+1} 个drug时发生错误：")
                print(f"错误类型：{type(e).__name__}")
                print(f"错误信息：{str(e)}")
                print("Drug数据：")
                print(drug_info)
                print("\n请检查上述数据中的问题")
                raise
        
        # 写入CSV文件
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"成功转换！输出CSV文件：{output_file}")
    
    except Exception as e:
        print(f"处理过程中发生错误：{str(e)}")
        import traceback
        traceback.print_exc()

# 使用示例
if __name__ == "__main__":
    input_json = r'C:\Users\86137\Desktop\Trial数据\self\drugbank\output.json'
    output_csv = r'C:\Users\86137\Desktop\Trial数据\self\drugbank\output.csv'
    
    # 测试模式：只处理前10个drug
    test_mode = False
    if test_mode:
        output_csv = r'C:\Users\86137\Desktop\Trial数据\self\drugbank\output_test.csv'
        json_to_csv(input_json, output_csv, max_drugs=10)
    else:
        # 正常模式：处理所有drug
        json_to_csv(input_json, output_csv)