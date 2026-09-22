"""查询改写后的两路召回：改写句 + 原句各搜一遍，RRF 融合。

为什么两路：LLM 改写可能翻车（丢掉关键数字/专有名词），原句召回可以兜底。
融合用 RRF（按名次取倒数加和）——与 Qdrant 服务端混合检索同一套数学。
"""

from langchain_core.documents import Document

from app.core.config import get_settings
from app.storage.vector_store import search

# 每路先多召回一些，给融合留空间
_CANDIDATE_K = 10
_RRF_K = 60


def rrf_fuse(lists: list[list[Document]], top_n: int = 5) -> list[Document]:
    """RRF 融合多路检索结果：只按名次打分，名次越前分越高。"""
    scores: dict[str, tuple[float, Document]] = {}
    for docs in lists:
        for rank, doc in enumerate(docs):
            key = doc.page_content[:100]  # 简易身份：同一 chunk 内容相同视为同一文档
            score, _ = scores.get(key, (0.0, doc))
            scores[key] = (score + 1.0 / (_RRF_K + rank + 1), doc)
    ranked = sorted(scores.values(), key=lambda item: -item[0])
    return [doc for _, doc in ranked[:top_n]]


def search_with_rewrite(rewritten: str, original: str, kb_id: str, top_k: int = 5) -> list[Document]:
    """改写句 + 原句两路召回 → RRF 融合 → Top-K。"""
    top_k = top_k or get_settings().retrieval_top_k
    docs_rewritten = search(kb_id, rewritten, k=_CANDIDATE_K)
    if rewritten == original:
        return docs_rewritten[:top_k]
    docs_original = search(kb_id, original, k=_CANDIDATE_K)
    return rrf_fuse([docs_rewritten, docs_original], top_n=top_k)
