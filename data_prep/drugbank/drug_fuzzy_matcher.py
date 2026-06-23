import pandas as pd
import numpy as np
import time
import re
from typing import List, Dict, Any, Optional, Tuple
from functools import lru_cache
import os

from fuzzywuzzy import fuzz
USING_FUZZYWUZZY = True


class DrugMatcher:
    """药物名称和商品名匹配工具"""
    
    def __init__(self, csv_path: str, preprocess: bool = True, use_synonyms: bool = False):
        """初始化药物匹配器
        
        Args:
            csv_path: CSV文件路径
            preprocess: 是否预处理数据以提高性能
            use_synonyms: 是否使用同义词列表进行匹配
        """
        self.use_synonyms = use_synonyms
        start_time = time.time()
        print(f"正在加载CSV文件: {csv_path}")
        
        # 检查文件是否存在
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"文件不存在: {csv_path}")
        
        # 读取CSV文件
        self.df = pd.read_csv(csv_path)
        
        # 确保至少有两列数据
        if len(self.df.columns) < 2:
            raise ValueError("CSV文件必须至少包含两列: 药品名称和商品名称列表")
        
        # 确保前两列为字符串类型
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        
        # 检查是否有同义词列(第三列)
        self.has_synonyms = len(self.df.columns) > 2
        if self.has_synonyms and self.use_synonyms:
            self.df.iloc[:, 2] = self.df.iloc[:, 2].astype(str)
            print("已启用同义词匹配功能")
        
        # 预处理数据
        if preprocess:
            self._preprocess_data()
        
        load_time = time.time() - start_time
        print(f"CSV文件加载完成，共{len(self.df)}条记录，耗时{load_time:.2f}秒")
    
    def _preprocess_data(self):
        """预处理数据以加速匹配"""
        # 创建小写版本的列，避免重复转换
        self.df['drug_name_lower'] = self.df.iloc[:, 0].str.lower()
        
        # 分析商品名列表
        self._process_product_names()
        
        # 处理同义词列表
        if self.has_synonyms and self.use_synonyms:
            self._process_synonyms()
        
        # 创建词条化版本，用于高级匹配
        self.df['drug_name_tokens'] = self.df['drug_name_lower'].apply(self._tokenize)
        
        # 转换为列表，加速后续操作
        self.drug_names = self.df['drug_name_lower'].tolist()
        
        # 创建倒排索引，提高部分匹配性能
        self._create_inverted_index()
    
    def _process_product_names(self):
        """处理商品名列表"""
        print("正在处理商品名列表...")
        start_time = time.time()
        
        # 创建商品名索引
        self.product_indices = {}
        self.product_lists = []
        
        # 遍历每一行
        for idx, row in enumerate(self.df.iloc[:, 1]):
            if isinstance(row, str):
                # 检查是否是列表（逗号分隔）
                if ',' in row:
                    # 分割并清理商品名列表
                    products = [p.strip().lower() for p in row.split(',') if p.strip()]
                else:
                    # 单个商品名
                    products = [row.strip().lower()] if row.strip() else []
                
                # 存储该行的商品名列表
                self.product_lists.append(products)
                
                # 将每个商品名映射到行索引
                for product in products:
                    if product not in self.product_indices:
                        self.product_indices[product] = []
                    self.product_indices[product].append(idx)
            else:
                self.product_lists.append([])
        
        print(f"商品名列表处理完成，共{len(self.product_indices)}个商品名，耗时{time.time() - start_time:.2f}秒")
    
    def _process_synonyms(self):
        """处理同义词列表"""
        print("正在处理同义词列表...")
        start_time = time.time()
        
        # 创建同义词索引
        self.synonym_indices = {}
        self.synonym_lists = []
        
        # 遍历每一行
        for idx, row in enumerate(self.df.iloc[:, 2]):
            if isinstance(row, str):
                # 检查是否是列表（逗号分隔）
                if ',' in row:
                    # 分割并清理同义词列表
                    synonyms = [s.strip().lower() for s in row.split(',') if s.strip()]
                else:
                    # 单个同义词
                    synonyms = [row.strip().lower()] if row.strip() else []
                
                # 存储该行的同义词列表
                self.synonym_lists.append(synonyms)
                
                # 将每个同义词映射到行索引
                for synonym in synonyms:
                    if synonym not in self.synonym_indices:
                        self.synonym_indices[synonym] = []
                    self.synonym_indices[synonym].append(idx)
            else:
                self.synonym_lists.append([])
        
        print(f"同义词列表处理完成，共{len(self.synonym_indices)}个同义词，耗时{time.time() - start_time:.2f}秒")
    
    def _create_inverted_index(self):
        """创建倒排索引用于快速子字符串搜索"""
        self.token_index = {}
        
        # 为每个词条创建倒排索引
        for idx, tokens in enumerate(self.df['drug_name_tokens']):
            for token in tokens:
                if len(token) >= 3:  # 只索引长度>=3的词条
                    if token not in self.token_index:
                        self.token_index[token] = []
                    self.token_index[token].append(idx)
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """将文本分词
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 分词结果
        """
        # 移除特殊字符并分词
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return [word for word in text.split() if word]
    
    @staticmethod
    def _preprocess_query(query: str) -> Tuple[str, Optional[str]]:
        """预处理查询词，分离药名和剂量信息
        
        Args:
            query: 原始查询词
            
        Returns:
            Tuple[str, Optional[str]]: 处理后的药名和剂量信息
        """
        # 匹配常见剂量模式，如 10mg, 100ml, 50ug 等
        dose_pattern = r'\s+\d+\s*(?:mg|g|ml|l|mcg|ug|mg/ml|g/l|mg/g|iu|u|miu|units?|tabs?|caps?|ampule|vial|dose|μg|mcl)(?:\b|$)'
        # 另外匹配纯数字模式，如 Diosmin 600
        numeric_pattern = r'\s+\d+(?:\.\d+)?(?:\b|$)'
        
        # 尝试匹配剂量
        dose_match = re.search(dose_pattern, query, re.IGNORECASE)
        if dose_match:
            drug_name = query[:dose_match.start()]
            dose_info = dose_match.group().strip()
            return drug_name.strip(), dose_info
        
        # 尝试匹配纯数字
        numeric_match = re.search(numeric_pattern, query)
        if numeric_match:
            drug_name = query[:numeric_match.start()]
            dose_info = numeric_match.group().strip()
            return drug_name.strip(), dose_info
        
        # 没有剂量信息
        return query.strip(), None
    
    @lru_cache(maxsize=1000)
    def _get_similarity(self, str1: str, str2: str) -> float:
        """计算两个字符串的相似度，使用缓存提高性能
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            
        Returns:
            float: 相似度分数
        """
        # 如果两个字符串相同，直接返回1.0
        if str1 == str2:
            return 1.0
        
        # 如果使用fuzzywuzzy库
        if USING_FUZZYWUZZY:
            # 使用比例相似度
            ratio = fuzz.ratio(str1, str2) / 100.0
            # 使用部分比例相似度，更好地处理一个字符串是另一个子串的情况
            partial_ratio = fuzz.partial_ratio(str1, str2) / 100.0
            # 使用排序比例相似度，更好地处理单词顺序颠倒的情况
            token_sort_ratio = fuzz.token_sort_ratio(str1, str2) / 100.0
            
            # 计算连续字符匹配加分
            continuous_bonus = self._calculate_continuous_match(str1, str2)
            
            # 综合得分，偏向连续字符匹配和精确匹配
            final_score = 0.3 * ratio + 0.3 * partial_ratio + 0.1 * token_sort_ratio + 0.3 * continuous_bonus
            return min(final_score, 1.0)  # 确保不超过1.0
        
        # 如果没有安装fuzzywuzzy，使用改进的相似度计算
        # 计算连续字符匹配
        continuous_bonus = self._calculate_continuous_match(str1, str2)
        
        # 计算包含关系得分
        if str1 in str2:
            contain_score = len(str1) / len(str2)
        elif str2 in str1:
            contain_score = len(str2) / len(str1)
        else:
            contain_score = 0.0
        
        # 简单的字符重叠比率
        common_chars = set(str1) & set(str2)
        overlap_score = len(common_chars) / max(len(set(str1)), len(set(str2)))
        
        # 综合得分
        final_score = 0.4 * continuous_bonus + 0.4 * contain_score + 0.2 * overlap_score
        return min(final_score, 1.0)  # 确保不超过1.0
    
    def _calculate_continuous_match(self, str1: str, str2: str) -> float:
        """计算两个字符串之间的最长连续匹配，并返回一个归一化的分数
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            
        Returns:
            float: 连续匹配的归一化分数 (0-1)
        """
        # 如果有空字符串，返回0
        if not str1 or not str2:
            return 0.0
            
        # 找出最长公共子串
        m = len(str1)
        n = len(str2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i-1] == str2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
        
        # 归一化分数：最长公共子串长度 / 较短字符串长度
        return max_len / min(len(str1), len(str2))
    
    def match_drug_name(self, query: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """匹配药品名称
        
        Args:
            query: 查询词
            threshold: 相似度阈值
            
        Returns:
            Optional[Dict[str, Any]]: 匹配结果
        """
        # 预处理查询词，分离剂量信息
        drug_query, dose_info = self._preprocess_query(query)
        query_lower = drug_query.lower().strip()
        
        # 1. 精确匹配
        for idx, name in enumerate(self.drug_names):
            if query_lower == name:
                return {
                    'index': idx,
                    'match_type': '药品名精确匹配',
                    'similarity': 1.0,
                    'matched_term': name,
                    'row': self.df.iloc[idx].to_dict(),
                    'original_query': query,
                    'processed_query': drug_query,
                    'dose_info': dose_info
                }
        
        # 2. 模糊匹配
        best_idx = -1
        best_sim = 0.0
        
        # 随机采样以加速大型数据集的搜索
        if len(self.df) > 1000:
            sample_indices = np.random.choice(len(self.df), 1000, replace=False)
        else:
            sample_indices = range(len(self.df))
        
        for idx in sample_indices:
            sim = self._get_similarity(query_lower, self.drug_names[idx])
            if sim > threshold and sim > best_sim:
                best_idx = idx
                best_sim = sim
        
        if best_idx >= 0:
            return {
                'index': best_idx,
                'match_type': '药品名模糊匹配',
                'similarity': best_sim,
                'matched_term': self.df.iloc[best_idx, 0],
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': drug_query,
                'dose_info': dose_info
            }
        
        # 3. 完整性匹配
        # 对于"Diosminn"这种情况，检查是否是某个药名的一部分多了一个字符
        for idx, name in enumerate(self.drug_names):
            if len(name) > 3 and (name in query_lower or 
                                (abs(len(name) - len(query_lower)) <= 2 and 
                                 self._calculate_continuous_match(name, query_lower) > 0.8)):
                sim = self._calculate_continuous_match(name, query_lower)
                if sim > threshold and sim > best_sim:
                    best_idx = idx
                    best_sim = sim
        
        if best_idx >= 0:
            return {
                'index': best_idx,
                'match_type': '药品名完整性匹配',
                'similarity': best_sim,
                'matched_term': self.df.iloc[best_idx, 0],
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': drug_query,
                'dose_info': dose_info
            }
        
        return None
    
    def match_product_name(self, query: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """匹配商品名
        
        Args:
            query: 查询词
            threshold: 相似度阈值
            
        Returns:
            Optional[Dict[str, Any]]: 匹配结果
        """
        # 预处理查询词，分离剂量信息
        product_query, dose_info = self._preprocess_query(query)
        query_lower = product_query.lower().strip()
        
        # 1. 精确匹配
        if query_lower in self.product_indices:
            idx = self.product_indices[query_lower][0]  # 取第一个匹配项
            return {
                'index': idx,
                'match_type': '商品名精确匹配',
                'similarity': 1.0,
                'matched_term': query_lower,
                'row': self.df.iloc[idx].to_dict(),
                'original_query': query,
                'processed_query': product_query,
                'dose_info': dose_info
            }
        
        # 2. 模糊匹配
        best_idx = -1
        best_sim = 0.0
        best_product = ""
        
        # 对于每个可能的商品名，计算相似度
        for product, indices in self.product_indices.items():
            sim = self._get_similarity(query_lower, product)
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_idx = indices[0]  # 取第一个匹配的行索引
                best_product = product
        
        # 如果找到了匹配，返回结果
        if best_idx >= 0:
            return {
                'index': best_idx,
                'match_type': '商品名模糊匹配',
                'similarity': best_sim,
                'matched_term': best_product,
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': product_query,
                'dose_info': dose_info
            }
        
        # 3. 词序颠倒匹配
        tokens = self._tokenize(query_lower)
        if len(tokens) > 1:
            # 生成词序颠倒的所有可能组合
            from itertools import permutations
            for perm in permutations(tokens):
                perm_query = ' '.join(perm)
                for product, indices in self.product_indices.items():
                    sim = self._get_similarity(perm_query, product)
                    if sim > threshold and sim > best_sim:
                        best_sim = sim
                        best_idx = indices[0]
                        best_product = product
            
            # 如果找到了匹配，返回结果
            if best_idx >= 0:
                return {
                    'index': best_idx,
                    'match_type': '商品名词序颠倒匹配',
                    'similarity': best_sim,
                    'matched_term': best_product,
                    'row': self.df.iloc[best_idx].to_dict(),
                    'original_query': query,
                    'processed_query': product_query,
                    'dose_info': dose_info
                }
        
        return None
    
    def match_synonym(self, query: str, threshold: float = 0.7) -> Optional[Dict[str, Any]]:
        """匹配同义词
        
        Args:
            query: 查询词
            threshold: 相似度阈值
            
        Returns:
            Optional[Dict[str, Any]]: 匹配结果
        """
        if not self.has_synonyms or not self.use_synonyms:
            return None
        
        # 预处理查询词，分离剂量信息
        synonym_query, dose_info = self._preprocess_query(query)
        query_lower = synonym_query.lower().strip()
        
        # 1. 精确匹配
        if query_lower in self.synonym_indices:
            idx = self.synonym_indices[query_lower][0]  # 取第一个匹配项
            return {
                'index': idx,
                'match_type': '同义词精确匹配',
                'similarity': 1.0,
                'matched_term': query_lower,
                'row': self.df.iloc[idx].to_dict(),
                'original_query': query,
                'processed_query': synonym_query,
                'dose_info': dose_info
            }
        
        # 2. 模糊匹配
        best_idx = -1
        best_sim = 0.0
        best_synonym = ""
        
        # 对于每个可能的同义词，计算相似度
        for synonym, indices in self.synonym_indices.items():
            sim = self._get_similarity(query_lower, synonym)
            if sim > threshold and sim > best_sim:
                best_sim = sim
                best_idx = indices[0]  # 取第一个匹配的行索引
                best_synonym = synonym
        
        # 如果找到了匹配，返回结果
        if best_idx >= 0:
            return {
                'index': best_idx,
                'match_type': '同义词模糊匹配',
                'similarity': best_sim,
                'matched_term': best_synonym,
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': synonym_query,
                'dose_info': dose_info
            }
        
        return None
    
    def search(self, query: str) -> Dict[str, Any]:
        """搜索匹配的药品
        
        Args:
            query: 查询词
            
        Returns:
            Dict[str, Any]: 搜索结果
        """
        if not query or not isinstance(query, str):
            return {
                "success": False,
                "message": "无效的查询词",
                "query": query
            }
        
        start_time = time.time()
        
        # 1. 尝试药品名匹配
        drug_result = self.match_drug_name(query)
        
        # 2. 尝试商品名匹配
        product_result = self.match_product_name(query)
        
        # 3. 尝试同义词匹配
        synonym_result = None
        if self.has_synonyms and self.use_synonyms:
            synonym_result = self.match_synonym(query)
        
        # 选择最佳匹配结果
        best_result = None
        best_sim = 0.0
        
        # 比较各种匹配结果，选择相似度最高的
        if drug_result and drug_result['similarity'] > best_sim:
            best_result = drug_result
            best_sim = drug_result['similarity']
        
        if product_result and product_result['similarity'] > best_sim:
            best_result = product_result
            best_sim = product_result['similarity']
        
        if synonym_result and synonym_result['similarity'] > best_sim:
            best_result = synonym_result
            best_sim = synonym_result['similarity']
        
        search_time = time.time() - start_time
        
        # 构建最终结果
        if best_result:
            return {
                "success": True,
                "result": best_result,
                "message": f"找到{best_result['match_type']}",
                "query": query,
                "search_time": search_time
            }
        else:
            return {
                "success": False,
                "message": "未找到匹配项",
                "query": query,
                "search_time": search_time
            }


def main():
    """主函数，提供交互式界面"""
    print("=" * 50)
    print("药品名称匹配工具")
    print("=" * 50)
    
    # 获取CSV文件路径
    csv_path = input("请输入CSV文件路径 (默认为drugbank_data_v1.csv): ").strip()
    if not csv_path:
        csv_path = "drugbank_data_v1.csv"
    
    # 询问是否启用同义词匹配
    use_synonyms = input("是否启用同义词匹配? (y/n, 默认为n): ").strip().lower() == 'y'
    
    try:
        # 初始化匹配器
        matcher = DrugMatcher(csv_path, use_synonyms=use_synonyms)
        
        print("\n" + "=" * 50)
        print("匹配器初始化完成，可以开始搜索")
        print("输入'q'或'quit'退出程序")
        print("输入'h'或'help'查看帮助")
        print("=" * 50)
        
        while True:
            # 获取用户输入
            query = input("\n请输入要搜索的药物名称: ").strip()
            
            # 检查退出命令
            if query.lower() in ['q', 'quit']:
                print("感谢使用，再见!")
                break
                
            # 检查帮助命令
            if query.lower() in ['h', 'help']:
                print("\n使用说明:")
                print("1. 直接输入药品名称或商品名进行搜索")
                print("2. 支持模糊匹配，如'aet's pirfenidone'可匹配'Pirfenidone Aet'")
                print("3. 支持商品名列表中的任意商品名")
                print("4. 如果启用了同义词匹配，还可以匹配同义词")
                print("5. 输入'q'或'quit'退出程序")
                print("6. 输入'h'或'help'查看此帮助")
                continue
            
            # 执行搜索
            if query:
                result = matcher.search(query)
                
                # 打印结果
                if result["success"]:
                    print(f"\n成功找到匹配项 ({result['message']}):")
                    print(f"查询词: {result['query']}")
                    print(f"处理后查询词: {result['result'].get('processed_query', result['query'])}")
                    if result['result'].get('dose_info'):
                        print(f"剂量信息: {result['result']['dose_info']}")
                    print(f"匹配词: {result['result']['matched_term']}")
                    print(f"相似度: {result['result']['similarity']:.2f}")
                    print(f"匹配类型: {result['result']['match_type']}")
                    print(f"行索引: {result['result']['index']}")
                    
                    # 打印药品信息
                    row = result['result']['row']
                    col_names = matcher.df.columns.tolist()
                    print(f"\n药品名({col_names[0]}): {row[col_names[0]]}")
                    print(f"商品名({col_names[1]}): {row[col_names[1]]}")
                else:
                    print(f"\n未找到匹配项: {result['message']}")
                    print(f"查询词: {result['query']}")
                
                print(f"\n查询耗时: {result['search_time']:.4f}秒")
            else:
                print("请输入有效的搜索词")
    
    except Exception as e:
        print(f"错误: {str(e)}")
        print("程序异常退出")


if __name__ == "__main__":
    main() 