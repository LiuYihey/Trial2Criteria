import xml.etree.ElementTree as ET
import json

def element_to_dict(element):
    """递归将XML元素转换为字典"""
    node_data = {}
    
    # 处理元素属性
    if element.attrib:
        node_data["@attributes"] = element.attrib
    
    # 处理文本内容
    text_content = element.text.strip() if element.text else ""
    if text_content:
        node_data["#text"] = text_content
    
    # 处理子元素
    for child in element:
        child_data = element_to_dict(child)
        
        # 处理重复节点
        if child.tag in node_data:
            # 如果已存在相同标签，转换为列表
            if not isinstance(node_data[child.tag], list):
                node_data[child.tag] = [node_data[child.tag]]
            node_data[child.tag].append(child_data)
        else:
            node_data[child.tag] = child_data
    
    # 处理空元素的文本表示
    if not text_content and not element.attrib and not node_data:
        return None  # 完全空元素返回None
    
    return node_data

def xml_to_json(xml_file, json_file):
    """主转换函数"""
    try:
        # 解析XML并获取根元素
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 构建字典结构
        xml_dict = {root.tag: element_to_dict(root)}
        
        # 生成JSON并保存
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(xml_dict, f, indent=2, ensure_ascii=False)
            
        return True
    except ET.ParseError as e:
        print(f"XML解析错误: {str(e)}")
        return False
    except Exception as e:
        print(f"转换失败: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    input_xml = "full_database.xml"
    output_json = "output.json"
    
    if xml_to_json(input_xml, output_json):
        print(f"转换成功，已保存到 {output_json}")
    else:
        print("转换失败")