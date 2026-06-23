import os
import pandas as pd
import numpy as np
import faiss
import pickle
import torch
from trial2vec import Trial2Vec

class TrialEncoder:
    def __init__(self, csv_path, model_path=None, index_path="trial_vectors.index", id_map_path="trial_id_map.pkl"):
        """
        初始化编码器
        
        参数:
        csv_path: CSV文件路径
        model_path: 预训练模型路径，默认为None（使用默认路径）
        index_path: FAISS索引保存路径
        id_map_path: ID映射保存路径
        """
        self.csv_path = csv_path
        self.model_path = model_path
        self.index_path = index_path
        self.id_map_path = id_map_path
        self.model = None
        self.embeddings = None
        self.trial_df = None
        self.index = None
        self.id_map = None
        
        # 检查CUDA是否可用
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        print(f"使用设备: {self.device}")
    
    def load_data(self):
        """加载并预处理CSV数据"""
        print(f"加载数据: {self.csv_path}")
        self.trial_df = pd.read_csv(self.csv_path)
        
        # 检查并重命名必要的列
        column_mapping = {
            # 以下是Trial2Vec期望的列名映射，根据您的数据调整
            # 'nct_number': 'nct_id',
            # 'official_title': 'title',
            # 'brief_summary': 'description',
            # 'intervention_type': 'intervention_name',
            # 'condition': 'disease',
            # 'eligibility_criteria': 'criteria'
        }
        
        # 应用列名映射
        for old_col, new_col in column_mapping.items():
            if old_col in self.trial_df.columns and new_col not in self.trial_df.columns:
                self.trial_df[new_col] = self.trial_df[old_col]
        
        # 检查必要的列是否存在
        required_fields = ['nct_id', 'title', 'description', 'intervention_name', 'disease', 'keyword', 'criteria']
        missing_fields = [field for field in required_fields if field not in self.trial_df.columns]
        
        if missing_fields:
            print(f"警告: 缺少以下字段: {missing_fields}")
            print("请确保CSV文件包含以下字段或适当映射: nct_id, title, description, intervention_name, disease, keyword, criteria")
            return False
        
        # 填充缺失值
        self.trial_df.fillna('', inplace=True)
        print(f"成功加载 {len(self.trial_df)} 条试验数据")
        return True
    
    def load_model(self):
        """加载Trial2Vec模型"""
        print("加载Trial2Vec模型...")
        self.model = Trial2Vec(device=self.device)
        
        if self.model_path:
            self.model.from_pretrained(self.model_path)
        else:
            # 使用默认路径
            try:
                self.model.from_pretrained('./trial_search/pretrained_trial2vec')
            except:
                print("未找到预训练模型，请确保您已下载Trial2Vec预训练模型")
                return False
        
        print("模型加载成功")
        return True
    
    def encode_trials(self):
        """编码试验数据"""
        print("开始编码试验数据...")
        
        # 准备输入数据
        input_data = {
            'x': self.trial_df,
            'fields': ['title', 'intervention_name', 'disease', 'keyword'],  # 属性字段
            'ctx_fields': ['description', 'criteria'],  # 上下文字段
            'tag': 'nct_id'  # 用于标识trial的字段
        }
        
        # 编码
        batch_size = 16 if self.device.startswith('cuda') else 8
        self.embeddings = self.model.encode(
            inputs=input_data,
            batch_size=batch_size,
            num_workers=4,
            return_dict=True,
            verbose=True
        )
        
        print(f"成功编码 {len(self.embeddings)} 个试验")
        return True
    
    def create_faiss_index(self):
        """创建FAISS索引"""
        print("创建FAISS索引...")
        
        # 将嵌入向量转换为numpy数组
        ids = list(self.embeddings.keys())
        vectors = np.array(list(self.embeddings.values()), dtype=np.float32)
        dimension = vectors.shape[1]  # 向量维度
        
        # 创建索引
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(vectors)
        
        # 创建ID映射
        self.id_map = {i: id for i, id in enumerate(ids)}
        
        print(f"FAISS索引创建完成，包含 {self.index.ntotal} 个向量")
        return True
    
    def save_index(self):
        """保存FAISS索引和ID映射"""
        print(f"保存索引到 {self.index_path}")
        faiss.write_index(self.index, self.index_path)
        
        print(f"保存ID映射到 {self.id_map_path}")
        with open(self.id_map_path, "wb") as f:
            pickle.dump(self.id_map, f)
        
        # 也保存嵌入向量以便后续使用
        with open("trial_embeddings.pkl", "wb") as f:
            pickle.dump(self.embeddings, f)
        
        print("索引和映射保存完成")
        return True
    
    def load_index(self):
        """加载FAISS索引和ID映射"""
        if not os.path.exists(self.index_path) or not os.path.exists(self.id_map_path):
            print("索引文件不存在，请先运行编码和索引创建")
            return False
        
        print(f"加载索引: {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        print(f"加载ID映射: {self.id_map_path}")
        with open(self.id_map_path, "rb") as f:
            self.id_map = pickle.load(f)
        
        print("索引加载完成")
        return True
    
    def search_similar_trials(self, query_nct_id, top_k=10):
        """根据NCT ID查找相似试验"""
        if self.index is None or self.id_map is None:
            success = self.load_index()
            if not success:
                return {"error": "无法加载索引"}
        
        # 检查是否有嵌入向量
        try:
            with open("trial_embeddings.pkl", "rb") as f:
                self.embeddings = pickle.load(f)
        except:
            pass
        
        # 获取查询向量
        if self.embeddings and query_nct_id in self.embeddings:
            query_vector = self.embeddings[query_nct_id].reshape(1, -1)
        else:
            # 如果嵌入向量未缓存，尝试从原始数据编码
            if self.trial_df is None:
                success = self.load_data()
                if not success:
                    return {"error": "无法加载数据"}
            
            if self.model is None:
                success = self.load_model()
                if not success:
                    return {"error": "无法加载模型"}
            
            query_trial = self.trial_df[self.trial_df['nct_id'] == query_nct_id]
            if len(query_trial) == 0:
                return {"error": f"找不到NCT ID: {query_nct_id}"}
            
            query_data = {
                'x': query_trial,
                'fields': ['title', 'intervention_name', 'disease', 'keyword'],
                'ctx_fields': ['description', 'criteria'],
                'tag': 'nct_id'
            }
            query_embedding = self.model.encode(query_data)
            query_vector = query_embedding[query_nct_id].reshape(1, -1)
        
        # 执行搜索
        distances, indices = self.index.search(query_vector, top_k)
        
        # 格式化结果
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.id_map):
                similar_nct_id = self.id_map[idx]
                similarity_score = float(1.0 / (1.0 + distances[0][i]))  # 转换距离为相似度分数
                
                # 获取试验详情
                trial_details = {}
                if self.trial_df is not None:
                    trial_row = self.trial_df[self.trial_df['nct_id'] == similar_nct_id]
                    if len(trial_row) > 0:
                        trial_details = {
                            "title": trial_row['title'].values[0],
                            "disease": trial_row['disease'].values[0]
                        }
                
                result = {
                    "nct_id": similar_nct_id,
                    "similarity_score": similarity_score,
                    "details": trial_details
                }
                results.append(result)
        
        return results
    
    def process_pipeline(self):
        """执行完整的编码、索引创建和保存流程"""
        success = self.load_data()
        if not success:
            return False
        
        success = self.load_model()
        if not success:
            return False
        
        success = self.encode_trials()
        if not success:
            return False
        
        success = self.create_faiss_index()
        if not success:
            return False
        
        success = self.save_index()
        if not success:
            return False
        
        print("编码和索引创建完成！")
        return True


def run_interactive_ui(encoder):
    """运行交互式用户界面"""
    import time
    
    print("\n" + "="*50)
    print("       临床试验相似性搜索系统")
    print("="*50)
    
    # 检查索引是否已存在
    if os.path.exists(encoder.index_path) and os.path.exists(encoder.id_map_path):
        print("\n发现现有索引。您想要做什么？")
        print("1. 使用现有索引进行搜索")
        print("2. 重新编码和创建索引")
        choice = input("请选择 (1/2): ").strip()
        
        if choice == "1":
            encoder.load_index()
            encoder.load_data()  # 加载数据以便获取试验详情
        else:
            encoder.process_pipeline()
    else:
        print("\n未发现索引，将执行编码和索引创建流程...")
        encoder.process_pipeline()
    
    while True:
        print("\n" + "-"*50)
        print("菜单:")
        print("1. 通过NCT ID搜索相似试验")
        print("2. 列出数据集中的样本NCT ID")
        print("3. 重新创建索引")
        print("4. 退出")
        
        choice = input("\n请选择操作 (1-4): ").strip()
        
        if choice == "1":
            query_nct_id = input("请输入NCT ID: ").strip()
            top_k = input("返回结果数量 (默认10): ").strip()
            top_k = int(top_k) if top_k.isdigit() else 10
            
            print(f"\n正在搜索与 {query_nct_id} 相似的试验...")
            start_time = time.time()
            results = encoder.search_similar_trials(query_nct_id, top_k)
            search_time = time.time() - start_time
            
            if "error" in results:
                print(f"错误: {results['error']}")
                continue
            
            print(f"\n找到 {len(results)} 个相似试验 (用时 {search_time:.3f} 秒):")
            print("-"*80)
            print(f"{'NCT ID':<15} {'相似度':<10} {'疾病':<20} {'标题'}")
            print("-"*80)
            
            for i, trial in enumerate(results):
                details = trial['details']
                title = details.get('title', '')[:40] + '...' if details.get('title', '') else ''
                disease = details.get('disease', '')[:20] if details.get('disease', '') else ''
                
                print(f"{trial['nct_id']:<15} {trial['similarity_score']:.4f}    {disease:<20} {title}")
            
            print("-"*80)
            
        elif choice == "2":
            if encoder.trial_df is None:
                encoder.load_data()
            
            if encoder.trial_df is not None:
                sample_size = min(10, len(encoder.trial_df))
                samples = encoder.trial_df.sample(sample_size)
                
                print(f"\n随机 {sample_size} 个样本试验:")
                print("-"*80)
                print(f"{'NCT ID':<15} {'疾病':<25} {'标题'}")
                print("-"*80)
                
                for _, row in samples.iterrows():
                    nct_id = row['nct_id']
                    title = row['title'][:50] + '...' if len(row['title']) > 50 else row['title']
                    disease = row['disease'][:25] if len(row['disease']) > 25 else row['disease']
                    
                    print(f"{nct_id:<15} {disease:<25} {title}")
                
                print("-"*80)
            else:
                print("无法加载试验数据")
            
        elif choice == "3":
            print("\n重新创建索引...")
            encoder.process_pipeline()
            
        elif choice == "4":
            print("\n感谢使用！再见！")
            break
            
        else:
            print("无效选择，请重试")


if __name__ == "__main__":
    # 创建编码器实例
    encoder = TrialEncoder(
        csv_path="Phase3_RAG_trial_base_completed.csv", 
        model_path=None,  # 使用默认路径
        index_path="trial_vectors.index",
        id_map_path="trial_id_map.pkl"
    )
    
    # 运行交互式UI
    run_interactive_ui(encoder)