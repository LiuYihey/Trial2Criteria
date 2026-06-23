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
        """Initializes the drug matcher.
        
        Args:
            csv_path: Path to the CSV file.
            preprocess: Whether to preprocess data for performance.
            use_synonyms: Whether to use the synonym list for matching.
        """
        self.use_synonyms = use_synonyms
        start_time = time.time()
        print(f"--- [INFO] Loading CSV file from: {csv_path} ---")
        
        # Check if the file exists
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        # Read the CSV file
        self.df = pd.read_csv(csv_path)
        
        # Ensure at least two columns exist
        if len(self.df.columns) < 2:
            raise ValueError("CSV file must contain at least two columns: drug name and product name list.")
        
        # Ensure the first two columns are of string type
        self.df.iloc[:, 0] = self.df.iloc[:, 0].astype(str)
        self.df.iloc[:, 1] = self.df.iloc[:, 1].astype(str)
        
        # Check for a synonym column (third column)
        self.has_synonyms = len(self.df.columns) > 2
        if self.has_synonyms and self.use_synonyms:
            self.df.iloc[:, 2] = self.df.iloc[:, 2].astype(str)
            print("--- [INFO] Synonym matching has been enabled. ---")
        
        # Preprocess data
        if preprocess:
            self._preprocess_data()
        
        load_time = time.time() - start_time
        print(f"--- [SUCCESS] CSV file loaded. Found {len(self.df)} records in {load_time:.2f} seconds. ---")
    
    def _preprocess_data(self):
        """Preprocesses data to accelerate matching."""
        # Create lowercase versions of columns to avoid repeated conversions
        self.df['drug_name_lower'] = self.df.iloc[:, 0].str.lower()
        
        # Analyze the product name list
        self._process_product_names()
        
        # Process the synonym list
        if self.has_synonyms and self.use_synonyms:
            self._process_synonyms()
        
        # Create a tokenized version for advanced matching
        self.df['drug_name_tokens'] = self.df['drug_name_lower'].apply(self._tokenize)
        
        # Convert to a list to speed up subsequent operations
        self.drug_names = self.df['drug_name_lower'].tolist()
    
    def _process_product_names(self):
        """Processes the list of product names."""
        print("--- [INFO] Processing product name list... ---")
        start_time = time.time()
        
        # Create a product name index
        self.product_indices = {}
        self.product_lists = []
        
        # Iterate over each row
        for idx, row in enumerate(self.df.iloc[:, 1]):
            if isinstance(row, str):
                # Check if it is a list (comma-separated)
                if ',' in row:
                    # Split and clean the product name list
                    products = [p.strip().lower() for p in row.split(',') if p.strip()]
                else:
                    # Single product name
                    products = [row.strip().lower()] if row.strip() else []
                
                # Store the product name list for this row
                self.product_lists.append(products)
                
                # Map each product name to the row index
                for product in products:
                    if product not in self.product_indices:
                        self.product_indices[product] = []
                    self.product_indices[product].append(idx)
            else:
                self.product_lists.append([])
        
        print(f"--- [SUCCESS] Product name list processed. Found {len(self.product_indices)} unique product names in {time.time() - start_time:.2f} seconds. ---")
    
    def _process_synonyms(self):
        """Processes the synonym list."""
        print("--- [INFO] Processing synonym list... ---")
        start_time = time.time()
        
        # Create a synonym index
        self.synonym_indices = {}
        self.synonym_lists = []
        
        # Iterate over each row
        for idx, row in enumerate(self.df.iloc[:, 2]):
            if isinstance(row, str):
                # Check if it is a list (comma-separated)
                if ',' in row:
                    # Split and clean the synonym list
                    synonyms = [s.strip().lower() for s in row.split(',') if s.strip()]
                else:
                    # Single synonym
                    synonyms = [row.strip().lower()] if row.strip() else []
                
                # Store the synonym list for this row
                self.synonym_lists.append(synonyms)
                
                # Map each synonym to the row index
                for synonym in synonyms:
                    if synonym not in self.synonym_indices:
                        self.synonym_indices[synonym] = []
                    self.synonym_indices[synonym].append(idx)
            else:
                self.synonym_lists.append([])
        
        print(f"--- [SUCCESS] Synonym list processed. Found {len(self.synonym_indices)} unique synonyms in {time.time() - start_time:.2f} seconds. ---")
    
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenizes the text.
        
        Args:
            text: Input text
            
        Returns:
            List[str]: Tokenized result
        """
        # Remove special characters and tokenize
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
            str1: 查询词 (e.g., "zina")
            str2: 数据库中的词 (e.g., "Dyna-zina" or "Clobetasol Propionate")
            min_ratio: 最小长度比例
            
        Returns:
            bool: 是否满足条件
        """
        # 1. 基础检查
        if not str1 or not str2 or len(str1) > len(str2):
            return False
        
        # 2. 长度比例检查，避免无意义的短词匹配
        if len(str1) / len(str2) < min_ratio:
            return False
        
        # 3. 边界词匹配 (最终修复)
        # 目标: 将 "Dyna-zina" 视为一个单词，而 "Clobetasol Propionate" 视为两个。
        # 做法: 只按空格分割数据库词条，并进行全词匹配。
        
        query_word = str1.lower()
        db_words = str2.lower().split() # 只按空格分割
        
        # 检查查询词是否与分割后的某个单词完全相等
        if query_word in db_words:
            return True
            
        return False
    
    @staticmethod
    def _calculate_main_drug_similarity(main1: str, main2: str) -> float:
        """计算两个主药名的相似度
        
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
        """搜索匹配的药品, with intelligent fallback for multi-word queries.
        
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
        
        # Try drug name match first
        drug_result = self.match_drug_name(query)
        if drug_result:
            drug_result['match_source'] = '药品名'
            return {
                "success": True,
                "result": drug_result,
                "message": f"Found {drug_result['match_type']}",
                "query": query,
                "search_time": time.time() - start_time
            }

        # Then try product name
        product_result = self.match_product_name(query)
        if product_result:
            product_result['match_source'] = '商品名'
            return {
                "success": True,
                "result": product_result,
                "message": f"Found {product_result['match_type']}",
                "query": query,
                "search_time": time.time() - start_time
            }
        
        # Finally, try synonym
        if self.has_synonyms and self.use_synonyms:
            synonym_result = self.match_synonym(query)
            if synonym_result:
                synonym_result['match_source'] = '同义词'
                return {
                    "success": True,
                    "result": synonym_result,
                    "message": f"Found {synonym_result['match_type']}",
                    "query": query,
                    "search_time": time.time() - start_time
                }

        # If all fail, return failure
        return {
            "success": False,
            "message": "未找到匹配项",
            "query": query,
            "search_time": time.time() - start_time
        }


def main():
    """主函数，提供交互式界面"""
    print("=" * 50)
    print("药品名称匹配工具")
    print("=" * 50)
    
    # 获取CSV文件路径
    csv_path = input("请输入CSV文件路径 (默认为drugbank_data_v1.csv): ").strip()
    if not csv_path:
        csv_path = "drug\drugbank_data_v1.csv"
    
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