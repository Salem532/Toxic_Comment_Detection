import re

def clean_text(text):
    """
    文本清洗：统一转小写 + 去除非字母字符 + 分词
    Args:
        text: 原始文本字符串
    Returns:
        清洗后的单词列表（list[str]）
    """
    # 转小写（避免Hello/hello/HELLO被视为不同词汇）
    text = str(text).lower()
    # 正则过滤：仅保留小写字母和空格，去除其他所有字符（符号、数字、表情等）
    text = re.sub(r'[^a-z\s]', '', text)
    # 按空格分词，返回单词列表（英文天然空格分隔，适合基础场景）
    return text.split()
