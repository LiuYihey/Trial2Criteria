import os
import sys
import json
from advanced_disease_extractor import AdvancedDiseaseExtractor

def main():
    """测试改进后的疾病提取器的两步API流程"""
    # 检查API密钥
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误: 未设置DEEPSEEK_API_KEY环境变量")
        print("请先设置此环境变量再运行测试")
        sys.exit(1)
    
    # 初始化提取器
    extractor = AdvancedDiseaseExtractor(api_key=api_key)
    
    # 测试查询
    test_queries = [
        "type 2 diabetes",
        "alzheimer's disease",
        "coronary artery disease"
    ]
    
    print("\n===== 测试新的两步式疾病匹配流程 =====\n")
    
    for query in test_queries:
        print(f"\n正在测试: '{query}'")
        result = extractor.get_info(query)
        
        if result:
            print(f"✓ 成功! 最佳匹配: {result.get('best_match_name', '未知')}")
            # 打印部分检索到的特征
            feature_count = 0
            for key, value in result.items():
                if key != 'best_match_name' and feature_count < 3:
                    print(f"  - {key}: {value[:2] if isinstance(value, list) and len(value) > 2 else value}")
                    feature_count += 1
        else:
            print(f"✗ 未能获取'{query}'的信息")
    
if __name__ == "__main__":
    main() 