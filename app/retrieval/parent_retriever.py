"""父子检索：子 chunk 命中后，回查并返回对应的父文档。

检索流程：混合检索命中子 chunk → 收集 parent_id → docstore 取回父文档
→ 按子 chunk 的排名顺序去重（同一父只取一次）→ 父文档列表交给下游
（rerank/judge/generate 读的是完整段落而非碎片）。
"""

import logging

from langchain_core.documents import Document

from app.storage.docstore import get_parents

logger = logging.getLogger(__name__)


def retrieve_parents(kb_id: str, child_docs: list[Document], top_n: int = 5) -> list[Document]:
    """子查父：按子 chunk 顺序收集父文档，去重后返回 Top-N。"""
    parent_ids: list[str] = []
    for doc in child_docs:
        pid = doc.metadata.get("parent_id")
        if pid:
            parent_ids.append(pid)
    if not parent_ids:
        logger.info("子 chunk 无 parent_id（旧数据），退回直接使用子 chunk")
        return child_docs[:top_n]

    parents = get_parents(kb_id, parent_ids)
    logger.info("父文档替换：%d 个子 chunk -> %d 个父文档", len(child_docs), len(parents))
    return parents[:top_n]
