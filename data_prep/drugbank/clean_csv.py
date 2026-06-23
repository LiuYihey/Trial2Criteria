import pandas as pd
import re

def is_garbled(text):
    if pd.isna(text):
        return False
    text = str(text)
    # 检测常见乱码模式
    patterns = [
        r'[\u4e00-\u9fff]',  # 中文字符
        r'[\u0080-\u009F]',  # 控制字符
        r'[\u00A0-\u00FF]',  # 扩展ASCII
        r'[\u0100-\u017F]',  # 拉丁扩展
        r'[\u0180-\u024F]',  # 拉丁扩展B
        r'[\u0250-\u02AF]',  # IPA扩展
        r'[\u02B0-\u02FF]',  # 修饰字母
        r'[\u0300-\u036F]',  # 组合用附加符号
        r'[\u0370-\u03FF]',  # 希腊字母
        r'[\u0400-\u04FF]',  # 西里尔字母
        r'[\u0500-\u052F]',  # 西里尔补充
        r'[\u0530-\u058F]',  # 亚美尼亚字母
        r'[\u0590-\u05FF]',  # 希伯来字母
        r'[\u0600-\u06FF]',  # 阿拉伯字母
        r'[\u0700-\u074F]',  # 叙利亚字母
        r'[\u0750-\u077F]',  # 阿拉伯补充
        r'[\u0780-\u07BF]',  # 马尔代夫字母
        r'[\u07C0-\u07FF]',  # 塔纳字母
        r'[\u0800-\u083F]',  # 撒玛利亚字母
        r'[\u0840-\u085F]',  # 曼达字母
        r'[\u0860-\u086F]',  # 叙利亚补充
        r'[\u0870-\u089F]',  # 阿拉伯扩展B
        r'[\u08A0-\u08FF]',  # 阿拉伯扩展A
        r'[\u0900-\u097F]',  # 天城文
        r'[\u0980-\u09FF]',  # 孟加拉文
        r'[\u0A00-\u0A7F]',  # 古木基文
        r'[\u0A80-\u0AFF]',  # 古吉拉特文
        r'[\u0B00-\u0B7F]',  # 奥里亚文
        r'[\u0B80-\u0BFF]',  # 泰米尔文
        r'[\u0C00-\u0C7F]',  # 泰卢固文
        r'[\u0C80-\u0CFF]',  # 卡纳达文
        r'[\u0D00-\u0D7F]',  # 马拉雅拉姆文
        r'[\u0D80-\u0DFF]',  # 僧伽罗文
        r'[\u0E00-\u0E7F]',  # 泰文
        r'[\u0E80-\u0EFF]',  # 老挝文
        r'[\u0F00-\u0FFF]',  # 藏文
        r'[\u1000-\u109F]',  # 缅甸文
        r'[\u10A0-\u10FF]',  # 格鲁吉亚文
        r'[\u1100-\u11FF]',  # 谚文
        r'[\u1200-\u137F]',  # 埃塞俄比亚文
        r'[\u1380-\u139F]',  # 埃塞俄比亚补充
        r'[\u13A0-\u13FF]',  # 切罗基文
        r'[\u1400-\u167F]',  # 统一加拿大原住民音节
        r'[\u1680-\u169F]',  # 欧甘文
        r'[\u16A0-\u16FF]',  # 如尼文
        r'[\u1700-\u171F]',  # 他加禄文
        r'[\u1720-\u173F]',  # 哈努诺文
        r'[\u1740-\u175F]',  # 布希德文
        r'[\u1760-\u177F]',  # 塔格班瓦文
        r'[\u1780-\u17FF]',  # 高棉文
        r'[\u1800-\u18AF]',  # 蒙古文
        r'[\u18B0-\u18FF]',  # 统一加拿大原住民音节扩展
        r'[\u1900-\u194F]',  # 林布文
        r'[\u1950-\u197F]',  # 德宏傣文
        r'[\u1980-\u19DF]',  # 新傣仂文
        r'[\u19E0-\u19FF]',  # 高棉符号
        r'[\u1A00-\u1A1F]',  # 布吉文
        r'[\u1A20-\u1AAF]',  # 老傣文
        r'[\u1AB0-\u1AFF]',  # 组合用附加符号扩展
        r'[\u1B00-\u1B7F]',  # 巴厘文
        r'[\u1B80-\u1BBF]',  # 巽他文
        r'[\u1BC0-\u1BFF]',  # 巴塔克文
        r'[\u1C00-\u1C4F]',  # 雷布查文
        r'[\u1C50-\u1C7F]',  # 桑塔利文
        r'[\u1C80-\u1C8F]',  # 西里尔扩展C
        r'[\u1C90-\u1CBF]',  # 格鲁吉亚扩展
        r'[\u1CC0-\u1CCF]',  # 巽他文补充
        r'[\u1CD0-\u1CFF]',  # 吠陀扩展
        r'[\u1D00-\u1D7F]',  # 音标扩展
        r'[\u1D80-\u1DBF]',  # 音标扩展补充
        r'[\u1DC0-\u1DFF]',  # 组合用附加符号补充
        r'[\u1E00-\u1EFF]',  # 拉丁扩展附加
        r'[\u1F00-\u1FFF]',  # 希腊扩展
        r'[\u2000-\u206F]',  # 常用标点
        r'[\u2070-\u209F]',  # 上标和下标
        r'[\u20A0-\u20CF]',  # 货币符号
        r'[\u20D0-\u20FF]',  # 组合用符号
        r'[\u2100-\u214F]',  # 字母式符号
        r'[\u2150-\u218F]',  # 数字形式
        r'[\u2190-\u21FF]',  # 箭头
        r'[\u2200-\u22FF]',  # 数学运算符
        r'[\u2300-\u23FF]',  # 杂项技术符号
        r'[\u2400-\u243F]',  # 控制图片
        r'[\u2440-\u245F]',  # 光学字符识别
        r'[\u2460-\u24FF]',  # 封闭式字母数字
        r'[\u2500-\u257F]',  # 制表符
        r'[\u2580-\u259F]',  # 方块元素
        r'[\u25A0-\u25FF]',  # 几何图形
        r'[\u2600-\u26FF]',  # 杂项符号
        r'[\u2700-\u27BF]',  # 装饰符号
        r'[\u27C0-\u27EF]',  # 杂项数学符号A
        r'[\u27F0-\u27FF]',  # 补充箭头A
        r'[\u2800-\u28FF]',  # 盲文图案
        r'[\u2900-\u297F]',  # 补充箭头B
        r'[\u2980-\u29FF]',  # 杂项数学符号B
        r'[\u2A00-\u2AFF]',  # 补充数学运算符
        r'[\u2B00-\u2BFF]',  # 杂项符号和箭头
        r'[\u2C00-\u2C5F]',  # 格拉哥里字母
        r'[\u2C60-\u2C7F]',  # 拉丁扩展C
        r'[\u2C80-\u2CFF]',  # 科普特字母
        r'[\u2D00-\u2D2F]',  # 格鲁吉亚补充
        r'[\u2D30-\u2D7F]',  # 提非纳文
        r'[\u2D80-\u2DDF]',  # 埃塞俄比亚扩展
        r'[\u2DE0-\u2DFF]',  # 西里尔扩展A
        r'[\u2E00-\u2E7F]',  # 补充标点
        r'[\u2E80-\u2EFF]',  # 中日韩部首补充
        r'[\u2F00-\u2FDF]',  # 康熙部首
        r'[\u2FF0-\u2FFF]',  # 表意文字描述符
        r'[\u3000-\u303F]',  # 中日韩符号和标点
        r'[\u3040-\u309F]',  # 平假名
        r'[\u30A0-\u30FF]',  # 片假名
        r'[\u3100-\u312F]',  # 注音字母
        r'[\u3130-\u318F]',  # 谚文兼容字母
        r'[\u3190-\u319F]',  # 象形文字注释
        r'[\u31A0-\u31BF]',  # 注音字母扩展
        r'[\u31C0-\u31EF]',  # 中日韩笔画
        r'[\u31F0-\u31FF]',  # 片假名扩展
        r'[\u3200-\u32FF]',  # 封闭式中日韩字母和月份
        r'[\u3300-\u33FF]',  # 中日韩兼容字符
        r'[\u3400-\u4DBF]',  # 中日韩统一表意文字扩展A
        r'[\u4DC0-\u4DFF]',  # 易经六十四卦符号
        r'[\u4E00-\u9FFF]',  # 中日韩统一表意文字
        r'[\uA000-\uA48F]',  # 彝文音节
        r'[\uA490-\uA4CF]',  # 彝文字根
        r'[\uA4D0-\uA4FF]',  # 老傈僳文
        r'[\uA500-\uA63F]',  # 瓦伊文
        r'[\uA640-\uA69F]',  # 西里尔扩展B
        r'[\uA6A0-\uA6FF]',  # 巴姆穆文
        r'[\uA700-\uA71F]',  # 声调修饰字母
        r'[\uA720-\uA7FF]',  # 拉丁扩展D
        r'[\uA800-\uA82F]',  # 锡尔赫特文
        r'[\uA830-\uA83F]',  # 通用印度数字形式
        r'[\uA840-\uA87F]',  # 八思巴文
        r'[\uA880-\uA8DF]',  # 索拉什特拉文
        r'[\uA8E0-\uA8FF]',  # 天城文扩展
        r'[\uA900-\uA92F]',  # 克耶文
        r'[\uA930-\uA95F]',  # 勒姜文
        r'[\uA960-\uA97F]',  # 谚文扩展A
        r'[\uA980-\uA9DF]',  # 爪哇文
        r'[\uA9E0-\uA9FF]',  # 缅甸文扩展B
        r'[\uAA00-\uAA5F]',  # 占文
        r'[\uAA60-\uAA7F]',  # 缅甸文扩展A
        r'[\uAA80-\uAADF]',  # 越南傣文
        r'[\uAAE0-\uAAFF]',  # 曼尼普尔文扩展
        r'[\uAB00-\uAB2F]',  # 埃塞俄比亚文扩展A
        r'[\uAB30-\uAB6F]',  # 拉丁扩展E
        r'[\uAB70-\uABBF]',  # 切罗基文补充
        r'[\uABC0-\uABFF]',  # 曼尼普尔文
        r'[\uAC00-\uD7AF]',  # 谚文音节
        r'[\uD7B0-\uD7FF]',  # 谚文兼容字母
        r'[\uD800-\uDB7F]',  # 高位代理
        r'[\uDB80-\uDBFF]',  # 高位代理
        r'[\uDC00-\uDFFF]',  # 低位代理
        r'[\uE000-\uF8FF]',  # 专用区
        r'[\uF900-\uFAFF]',  # 中日韩兼容表意文字
        r'[\uFB00-\uFB4F]',  # 字母表达形式
        r'[\uFB50-\uFDFF]',  # 阿拉伯表达形式A
        r'[\uFE00-\uFE0F]',  # 变体选择符
        r'[\uFE10-\uFE1F]',  # 竖排形式
        r'[\uFE20-\uFE2F]',  # 组合用半符号
        r'[\uFE30-\uFE4F]',  # 中日韩兼容形式
        r'[\uFE50-\uFE6F]',  # 小写变体
        r'[\uFE70-\uFEFF]',  # 阿拉伯表达形式B
        r'[\uFF00-\uFFEF]',  # 半角及全角形式
        r'[\uFFF0-\uFFFF]',  # 特殊
    ]
    
    for pattern in patterns:
        if re.search(pattern, text):
            return True
    return False

def clean_cell(cell):
    if is_garbled(cell):
        return "none"
    return cell

def main():
    print("开始读取CSV文件...")
    # 读取CSV文件
    df = pd.read_csv('output.csv', low_memory=False)
    
    print("开始清洗数据...")
    total_cells = df.size
    replaced_cells = 0
    
    # 对每一列应用清洗函数
    for column in df.columns:
        print(f"处理列: {column}")
        original_values = df[column].values
        df[column] = df[column].apply(clean_cell)
        replaced_cells += (original_values != df[column].values).sum()
    
    print(f"\n统计信息:")
    print(f"总单元格数: {total_cells}")
    print(f"替换单元格数: {replaced_cells}")
    print(f"替换比例: {(replaced_cells/total_cells)*100:.2f}%")
    
    print("\n保存清洗后的文件...")
    # 保存处理后的文件
    df.to_csv('output_cleaned.csv', index=False)
    print("处理完成！")

if __name__ == "__main__":
    main() 