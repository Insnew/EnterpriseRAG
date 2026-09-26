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
    """切分入口：表格原子化——chunk_kind="table" 的 Document 原样保留，永不切开。

    M6 起 Excel/PDF 表格会被 loader 标记为 chunk_kind="table"，
    在这里分流：表格直接放行，普通文本走 400 字切分。
    """
    tables = [d for d in docs if d.metadata.get("chunk_kind") == "table"]
    normal = [d for d in docs if d.metadata.get("chunk_kind") != "table"]
    return get_splitter().split_documents(normal) + tables
