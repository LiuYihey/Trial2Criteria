import pandas as pd
import time
import re
from typing import List, Dict, Any, Optional, Tuple
import os


class DrugMatcher:
    """药物名称和商品名匹配工具"""
    
    # 通用药品剂型和描述词列表
    generic_terms = [
        'oral capsule', 'capsule', 'tablet', 'oral tablet', 
        'injection', 'solution', 'suspension', 'fluid expansion',
        'oral solution', 'oral suspension', 'cream', 'ointment',
        'patch', 'suppository', 'spray', 'syrup', 'powder',
        'intravenous', 'intramuscular', 'subcutaneous', 'topical',
        'in water', 'for injection', 'in saline', 'iv'
    ]
    
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
        """预处理查询词，去除通用描述词并分离药名和剂量信息
        
        Args:
            query: 原始查询词
            
        Returns:
            Tuple[str, Optional[str]]: 处理后的药名和剂量信息
        """
        # 获取通用描述词列表
        generic_terms = DrugMatcher.generic_terms
        
        # 清理输入并转换为小写
        query = query.strip()
        query_lower = query.lower()
        
        # 移除通用描述词
        for term in generic_terms:
            # 检查药名后面跟通用描述词的情况
            term_pattern = r'\s+' + re.escape(term) + r'\b'
            query_lower = re.sub(term_pattern, '', query_lower)
            
            # 检查通用描述词在药名前面的情况
            term_pattern = r'\b' + re.escape(term) + r'\s+'
            query_lower = re.sub(term_pattern, '', query_lower)
        
        # 匹配常见剂量模式，如 10mg, 100ml, 50ug 等
        dose_pattern = r'\s+\d+\s*(?:mg|g|ml|l|mcg|ug|mg/ml|g/l|mg/g|iu|u|miu|units?|tabs?|caps?|ampule|vial|dose|μg|mcl)(?:\b|$)'
        # 另外匹配纯数字模式，如 Diosmin 600
        numeric_pattern = r'\s+\d+(?:\.\d+)?(?:\b|$)'
        
        # 尝试匹配剂量
        dose_match = re.search(dose_pattern, query_lower, re.IGNORECASE)
        if dose_match:
            drug_name = query_lower[:dose_match.start()]
            dose_info = dose_match.group().strip()
            return drug_name.strip(), dose_info
        
        # 尝试匹配纯数字
        numeric_match = re.search(numeric_pattern, query_lower)
        if numeric_match:
            drug_name = query_lower[:numeric_match.start()]
            dose_info = numeric_match.group().strip()
            return drug_name.strip(), dose_info
        
        # 没有剂量信息
        return query_lower.strip(), None
    
    @staticmethod
    def _normalize_for_comparison(text: str) -> str:
        """标准化文本以进行比较，处理's和-等
        
        Args:
            text: 输入文本
            
        Returns:
            str: 标准化后的文本
        """
        # 替换拥有格形式('s)为连字符
        text = re.sub(r"'s\b", "-", text.lower())
        # 将连字符也替换为空格，使Sodium-chloride和Sodium chloride能够匹配
        text = re.sub(r"-", " ", text)
        # 替换多个连续空格为单个空格
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    
    @staticmethod
    def _extract_drug_components(text: str) -> Tuple[str, str]:
        """将药品名称分解为品牌前缀和主要药物名称
        
        Args:
            text: 药品名称
            
        Returns:
            Tuple[str, str]: (品牌前缀, 主要药物名称)
        """
        # 尝试识别常见的分隔模式，如"PHL-oxybutynin"
        parts = re.split(r"[-\s]", text, 1)
        if len(parts) == 2:
            prefix = parts[0].strip()
            # 如果前缀是2-4个字符的纯字母，很可能是品牌前缀
            if 2 <= len(prefix) <= 4 and prefix.isalpha():
                return prefix, parts[1].strip()
        
        # 如果无法分解，返回空前缀和整个文本
        return "", text
    
    @staticmethod
    def _is_valid_completion_match(str1: str, str2: str, min_ratio: float = 0.3) -> bool:
        """判断两个字符串是否满足完整性匹配条件
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            min_ratio: 最小重叠比例
            
        Returns:
            bool: 是否满足完整性匹配条件
        """
        # 标准化字符串，处理's和-等
        str1_norm = DrugMatcher._normalize_for_comparison(str1)
        str2_norm = DrugMatcher._normalize_for_comparison(str2)
        
        # 如果标准化后完全相同，直接返回True
        if str1_norm == str2_norm:
            return True
        
        # 提取药品名称组件
        prefix1, main1 = DrugMatcher._extract_drug_components(str1_norm)
        prefix2, main2 = DrugMatcher._extract_drug_components(str2_norm)
        
        # 如果主要药物名称为空，不匹配
        if not main1 or not main2:
            return False
        
        # 检查主要药物名称是否匹配
        # 必须要求主要药物名称有显著重叠，这是最关键的部分
        main_similarity = DrugMatcher._calculate_main_drug_similarity(main1, main2)
        
        # 如果主要药物名称相似度不够高，直接返回False
        if main_similarity < 0.7:  # 要求主要药物名称至少70%的相似度
            return False
        
        # 检查前缀是否兼容
        prefix_compatible = True
        # 如果两者都有前缀且不同，则不匹配
        if prefix1 and prefix2 and prefix1 != prefix2:
            prefix_compatible = False
        
        # 检查词汇重叠
        # 清理特殊字符
        str1_clean = re.sub(r'[^\w\s]', ' ', str1_norm)
        str2_clean = re.sub(r'[^\w\s]', ' ', str2_norm)
        
        # 分词处理
        tokens1 = [t for t in str1_clean.split() if t]
        tokens2 = [t for t in str2_clean.split() if t]
        
        # 如果一个字符串为空，不匹配
        if not tokens1 or not tokens2:
            return False
        
        # 计算重叠词
        overlap_tokens = set(tokens1) & set(tokens2)
        
        # 计算重叠比例
        if not overlap_tokens:
            return False
            
        ratio1 = len(overlap_tokens) / len(tokens1)
        ratio2 = len(overlap_tokens) / len(tokens2)
        
        # 综合判断
        # 1. 主要药物名称相似度高
        # 2. 前缀兼容
        # 3. 整体有足够的词汇重叠
        if main_similarity >= 0.7 and prefix_compatible and (ratio1 >= min_ratio and ratio2 >= min_ratio):
            return True
            
        # 如果主要药物名称完全匹配，即使其他条件稍差也接受
        if main1 == main2:
            return ratio1 >= 0.5 or ratio2 >= 0.5
            
        return False
    
    @staticmethod
    def _calculate_main_drug_similarity(main1: str, main2: str) -> float:
        """计算主要药物名称的相似度
        
        Args:
            main1: 第一个主要药物名称
            main2: 第二个主要药物名称
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 如果完全相同，返回1.0
        if main1 == main2:
            return 1.0
            
        # 如果一个包含另一个，返回较高的相似度
        if main1 in main2:
            return len(main1) / len(main2)
        
        if main2 in main1:
            return len(main2) / len(main1)
        
        # 计算最长公共子序列
        tokens1 = main1.split()
        tokens2 = main2.split()
        
        # 如果有共同的词，计算共同词比例
        common_tokens = set(tokens1) & set(tokens2)
        if common_tokens:
            return len(common_tokens) / max(len(tokens1), len(tokens2))
            
        # 字符级别的最长公共子串
        m, n = len(main1), len(main2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        max_len = 0
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if main1[i-1] == main2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    max_len = max(max_len, dp[i][j])
        
        # 归一化
        return max_len / max(m, n)
    
    @staticmethod
    def _calculate_completion_score(str1: str, str2: str) -> float:
        """计算两个字符串的完整性匹配得分，用于排序候选匹配项
        
        Args:
            str1: 第一个字符串
            str2: 第二个字符串
            
        Returns:
            float: 匹配得分，值越高表示匹配质量越好
        """
        # 标准化字符串，处理's和-等
        str1_norm = DrugMatcher._normalize_for_comparison(str1)
        str2_norm = DrugMatcher._normalize_for_comparison(str2)
        
        # 如果标准化后完全相同，得分最高
        if str1_norm == str2_norm:
            return 1.0
        
        # 提取药品名称组件
        prefix1, main1 = DrugMatcher._extract_drug_components(str1_norm)
        prefix2, main2 = DrugMatcher._extract_drug_components(str2_norm)
        
        # 计算主要药物名称的相似度
        main_similarity = DrugMatcher._calculate_main_drug_similarity(main1, main2)
        
        # 计算前缀匹配得分
        prefix_score = 0.0
        if prefix1 and prefix2:
            # 如果前缀相同，得分最高
            if prefix1 == prefix2:
                prefix_score = 1.0
            # 如果前缀有部分重叠，得分相对较低
            elif prefix1 in prefix2 or prefix2 in prefix1:
                prefix_score = 0.5
        # 如果一个有前缀，一个没有，得分为0
        
        # 计算整体词汇重叠度
        str1_clean = re.sub(r'[^\w\s]', ' ', str1_norm)
        str2_clean = re.sub(r'[^\w\s]', ' ', str2_norm)
        
        tokens1 = [t for t in str1_clean.split() if t]
        tokens2 = [t for t in str2_clean.split() if t]
        
        overlap_tokens = set(tokens1) & set(tokens2)
        
        if not overlap_tokens:
            overall_score = 0.0
        else:
            ratio1 = len(overlap_tokens) / len(tokens1)
            ratio2 = len(overlap_tokens) / len(tokens2)
            overall_score = (ratio1 + ratio2) / 2
        
        # 字符串长度比例，越接近1越好
        len_ratio = min(len(str1_norm), len(str2_norm)) / max(len(str1_norm), len(str2_norm))
        
        # 综合计算最终得分，前缀匹配有较高权重
        # 主要药物名称相似度: 50%, 前缀匹配得分: 30%, 整体重叠度: 15%, 长度比例: 5%
        final_score = (0.5 * main_similarity) + (0.3 * prefix_score) + (0.15 * overall_score) + (0.05 * len_ratio)
        
        return final_score
    
    @staticmethod
    def _is_code_format(text: str) -> bool:
        """判断文本是否是代号格式（如NC-503, ADH503等）
        
        Args:
            text: 输入文本
            
        Returns:
            bool: 是否是代号格式
        """
        # 清理输入
        text = text.strip()
        
        # 代号模式: 2-4个字母 + 可选的连字符(-) + 3-4个数字
        # 例如: NC-503, ADH503
        pattern1 = r'^[A-Za-z]{2,4}-?\d{3,4}$'
        
        # 或者: 3-4个数字 + 可选的连字符(-) + 2-4个字母
        # 例如: 503-NC, 503ADH
        pattern2 = r'^\d{3,4}-?[A-Za-z]{2,4}$'
        
        return bool(re.match(pattern1, text) or re.match(pattern2, text))
    
    def match_drug_name(self, query: str) -> Optional[Dict[str, Any]]:
        """匹配药品名称
        
        Args:
            query: 查询词
            
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
        
        # 检查是否是代号格式，如果是则只允许精确匹配
        if self._is_code_format(query_lower):
            return None
        
        # 2. 完整性匹配
        # 使用字典记录所有可能的匹配和对应的分数
        candidate_matches = []
        
        for idx, name in enumerate(self.drug_names):
            if self._is_valid_completion_match(query_lower, name):
                # 计算匹配得分
                score = self._calculate_completion_score(query_lower, name)
                candidate_matches.append((idx, name, score))
        
        # 如果有候选匹配，选择得分最高的
        if candidate_matches:
            # 按得分降序排序
            candidate_matches.sort(key=lambda x: x[2], reverse=True)
            best_idx, best_match_term, best_score = candidate_matches[0]
            
            return {
                'index': best_idx,
                'match_type': '药品名完整性匹配',
                'similarity': 0.9,  # 完整性匹配给一个较高但不是1.0的相似度
                'matched_term': best_match_term,
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': drug_query,
                'dose_info': dose_info
            }
        
        return None
    
    def match_product_name(self, query: str) -> Optional[Dict[str, Any]]:
        """匹配商品名
        
        Args:
            query: 查询词
            
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
        
        # 检查是否是代号格式，如果是则只允许精确匹配
        if self._is_code_format(query_lower):
            return None
            
        # 2. 完整性匹配
        # 使用字典记录所有可能的匹配和对应的分数
        candidate_matches = []
        
        # 对于每个可能的商品名，检查完整性匹配
        for product, indices in self.product_indices.items():
            if self._is_valid_completion_match(query_lower, product):
                # 计算匹配得分
                score = self._calculate_completion_score(query_lower, product)
                candidate_matches.append((indices[0], product, score))
        
        # 如果有候选匹配，选择得分最高的
        if candidate_matches:
            # 按得分降序排序
            candidate_matches.sort(key=lambda x: x[2], reverse=True)
            best_idx, best_product, best_score = candidate_matches[0]
            
            return {
                'index': best_idx,
                'match_type': '商品名完整性匹配',
                'similarity': 0.9,  # 完整性匹配给一个较高但不是1.0的相似度
                'matched_term': best_product,
                'row': self.df.iloc[best_idx].to_dict(),
                'original_query': query,
                'processed_query': product_query,
                'dose_info': dose_info
            }
        
        return None
    
    def match_synonym(self, query: str) -> Optional[Dict[str, Any]]:
        """匹配同义词
        
        Args:
            query: 查询词
            
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
        
        # 检查是否是代号格式，如果是则只允许精确匹配
        if self._is_code_format(query_lower):
            return None
        
        # 2. 完整性匹配
        # 使用字典记录所有可能的匹配和对应的分数
        candidate_matches = []
        
        # 对于每个可能的同义词，检查完整性匹配
        for synonym, indices in self.synonym_indices.items():
            if self._is_valid_completion_match(query_lower, synonym):
                # 计算匹配得分
                score = self._calculate_completion_score(query_lower, synonym)
                candidate_matches.append((indices[0], synonym, score))
        
        # 如果有候选匹配，选择得分最高的
        if candidate_matches:
            # 按得分降序排序
            candidate_matches.sort(key=lambda x: x[2], reverse=True)
            best_idx, best_synonym, best_score = candidate_matches[0]
            
            return {
                'index': best_idx,
                'match_type': '同义词完整性匹配',
                'similarity': 0.9,  # 完整性匹配给一个较高但不是1.0的相似度
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
        
        # 预处理查询词，分离剂量信息
        product_query, _ = self._preprocess_query(query)
        query_lower = product_query.lower().strip()
        
        # 搜索所有列的匹配结果
        
        # 1. 搜索药品名精确匹配
        drug_exact = None
        for idx, name in enumerate(self.drug_names):
            if query_lower == name:
                drug_exact = {
                    'index': idx,
                    'match_type': '药品名精确匹配',
                    'similarity': 1.0,
                    'matched_term': name,
                    'row': self.df.iloc[idx].to_dict(),
                    'original_query': query,
                    'processed_query': product_query,
                    'match_source': '药品名'  # 添加匹配来源标记
                }
                break
        
        # 2. 搜索商品名精确匹配
        product_exact = None
        if query_lower in self.product_indices:
            idx = self.product_indices[query_lower][0]
            product_exact = {
                'index': idx,
                'match_type': '商品名精确匹配',
                'similarity': 1.0,
                'matched_term': query_lower,
                'row': self.df.iloc[idx].to_dict(),
                'original_query': query,
                'processed_query': product_query,
                'match_source': '商品名'  # 添加匹配来源标记
            }
        
        # 3. 搜索同义词精确匹配
        synonym_exact = None
        if self.has_synonyms and self.use_synonyms and query_lower in self.synonym_indices:
            idx = self.synonym_indices[query_lower][0]
            synonym_exact = {
                'index': idx,
                'match_type': '同义词精确匹配',
                'similarity': 1.0,
                'matched_term': query_lower,
                'row': self.df.iloc[idx].to_dict(),
                'original_query': query,
                'processed_query': product_query,
                'match_source': '同义词'  # 添加匹配来源标记
            }
        
        # 如果找到精确匹配，按固定顺序返回（药品名、商品名、同义词）
        exact_matches = []
        if drug_exact:
            exact_matches.append(drug_exact)
        if product_exact:
            exact_matches.append(product_exact)
        if synonym_exact:
            exact_matches.append(synonym_exact)
            
        if exact_matches:
            # 添加剂量信息
            _, dose_info = self._preprocess_query(query)
            
            # 所有精确匹配都有相同优先级，返回第一个找到的
            best_result = exact_matches[0]
            
            if dose_info:
                best_result['dose_info'] = dose_info
                
            search_time = time.time() - start_time
            return {
                "success": True,
                "result": best_result,
                "message": f"找到{best_result['match_type']}",
                "query": query,
                "search_time": search_time,
                "all_exact_matches": exact_matches,  # 提供所有精确匹配结果
                "exact_match_sources": {  # 添加匹配来源映射
                    "drug": drug_exact is not None,
                    "product": product_exact is not None,
                    "synonym": synonym_exact is not None
                }
            }
        
        # 如果没有精确匹配，尝试完整性匹配
        
        # 1. 尝试药品名完整性匹配
        drug_result = self.match_drug_name(query)
        if drug_result:
            drug_result['match_source'] = '药品名'  # 添加匹配来源标记
        
        # 2. 尝试商品名完整性匹配
        product_result = self.match_product_name(query)
        if product_result:
            product_result['match_source'] = '商品名'  # 添加匹配来源标记
        
        # 3. 尝试同义词完整性匹配
        synonym_result = None
        if self.has_synonyms and self.use_synonyms:
            synonym_result = self.match_synonym(query)
            if synonym_result:
                synonym_result['match_source'] = '同义词'  # 添加匹配来源标记
        
        # 收集所有完整性匹配结果，按固定顺序（药品名、商品名、同义词）
        completion_matches = []
        if drug_result:
            completion_matches.append(drug_result)
        if product_result:
            completion_matches.append(product_result)
        if synonym_result:
            completion_matches.append(synonym_result)
            
        search_time = time.time() - start_time
        
        # 构建最终结果
        if completion_matches:
            # 按相似度排序，选择最高相似度的结果
            completion_matches.sort(key=lambda x: x['similarity'], reverse=True)
            best_result = completion_matches[0]
            
            return {
                "success": True,
                "result": best_result,
                "message": f"找到{best_result['match_type']}",
                "query": query,
                "search_time": search_time,
                "all_completion_matches": completion_matches,  # 提供所有完整性匹配结果
                "completion_match_sources": {  # 添加匹配来源映射
                    "drug": drug_result is not None,
                    "product": product_result is not None,
                    "synonym": synonym_result is not None
                }
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
                print("2. 支持完整性匹配，例如:")
                print("   - Ditropan XL能够匹配Ditropan")
                print("   - PMS-oxybutynin能够匹配oxybutynin")
                print("   - XL Ditropan 能够匹配Ditropan XL")
                print("   - PHL's oxybutynin 能够匹配 PHL-oxybutynin")
                print("3. 支持商品名列表中的任意商品名")
                print("4. 如果启用了同义词匹配，还可以匹配同义词")
                print("5. 输入'q'或'quit'退出程序")
                print("6. 输入'h'或'help'查看此帮助")
                print("7. 注意: 对于代号格式(如NC-503)仅执行精确匹配")
                continue
            
            # 执行搜索
            if query:
                result = matcher.search(query)
                
                # 打印结果
                if result["success"]:
                    print(f"\n查询词: {result['query']}")
                    print(f"处理后查询词: {result['result'].get('processed_query', result['query'])}")
                    if result['result'].get('dose_info'):
                        print(f"剂量信息: {result['result']['dose_info']}")
                    
                    # 检查是否有多个精确匹配
                    if "all_exact_matches" in result and len(result["all_exact_matches"]) > 0:
                        print("\n找到精确匹配结果:")
                        
                        # 获取匹配来源映射
                        match_sources = {
                            match["match_source"]: i
                            for i, match in enumerate(result["all_exact_matches"])
                        }
                        
                        # 按固定顺序显示结果
                        source_labels = {"药品名": 1, "商品名": 2, "同义词": 3}
                        
                        # 对每个匹配类型，如果存在则显示
                        for source, label in source_labels.items():
                            if source in match_sources:
                                match = result["all_exact_matches"][match_sources[source]]
                                print(f"\n[精确匹配 #{label}] {source}")
                                print(f"匹配词: {match['matched_term']}")
                                print(f"匹配类型: {match['match_type']}")
                                print(f"相似度: {match['similarity']:.2f}")
                                
                                # 打印药品信息
                                row = match['row']
                                col_names = matcher.df.columns.tolist()
                                print(f"药品名({col_names[0]}): {row[col_names[0]]}")
                                print(f"商品名({col_names[1]}): {row[col_names[1]]}")
                    
                    # 检查是否有完整性匹配
                    elif "all_completion_matches" in result and len(result["all_completion_matches"]) > 0:
                        print("\n找到完整性匹配结果:")
                        
                        # 获取匹配来源映射
                        match_sources = {
                            match["match_source"]: i
                            for i, match in enumerate(result["all_completion_matches"])
                        }
                        
                        # 按固定顺序显示结果
                        source_labels = {"药品名": 1, "商品名": 2, "同义词": 3}
                        
                        # 对每个匹配类型，如果存在则显示
                        for source, label in source_labels.items():
                            if source in match_sources:
                                match = result["all_completion_matches"][match_sources[source]]
                                print(f"\n[完整性匹配 #{label}] {source}")
                                print(f"匹配词: {match['matched_term']}")
                                print(f"匹配类型: {match['match_type']}")
                                print(f"相似度: {match['similarity']:.2f}")
                                
                                # 打印药品信息
                                row = match['row']
                                col_names = matcher.df.columns.tolist()
                                print(f"药品名({col_names[0]}): {row[col_names[0]]}")
                                print(f"商品名({col_names[1]}): {row[col_names[1]]}")
                    
                    else:
                        print("\n未找到匹配项，但有最佳结果:")
                        print(f"匹配词: {result['result']['matched_term']}")
                        print(f"匹配类型: {result['result']['match_type']}")
                        print(f"相似度: {result['result']['similarity']:.2f}")
                        
                        # 打印药品信息
                        row = result['result']['row']
                        col_names = matcher.df.columns.tolist()
                        print(f"药品名({col_names[0]}): {row[col_names[0]]}")
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