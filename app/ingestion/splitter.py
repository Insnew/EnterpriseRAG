"""中文文档切分：RecursiveCharacterTextSplitter，优先按中文标点切分。

参数（400 字 + 80 字重叠）为中文文档经验值——M2 会用评测集做 chunk_size 扫描，
用数据决定最终参数，而不是拍脑袋。M6 加入表格原子化（chunk_kind="table" 永不切开）。
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 400
CHUNK_OVERLAP = 80
# 优先按段落/换行切，其次中文标点，最后空格兜底
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " "]


def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
    )


def split_documents(docs: list[Document]) -> list[Document]:
    return get_splitter().split_documents(docs)
