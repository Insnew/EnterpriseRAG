"""重排 API 客户端：bge-reranker-v2-m3 走硅基流动 OpenAI 兼容端点。

为什么直调 httpx 而不是 LangChain 类：LangChain 没有硅基流动的 rerank
集成包（CohereRerank 是 Cohere 家的）。直调最透明，~40 行。

⚠️ 已知坑（面试可讲）：硅基流动返回的 relevance_score 是归一化后的极小值
（真正相关的通常也只有 ~0.003），且跨查询不可比——所以分数只用于
【排序】，绝不能拿去做"相关/不相关"的阈值判断。拒答判断交给 LLM judge
（M5 图节点），本文件只负责"打分排序"这一件事。
"""

import logging

import httpx

from app.core.config import get_settings

logger = logging.getLogger(__name__)

RERANK_MODEL = "BAAI/bge-reranker-v2-m3"


def rerank(query: str, documents: list[str], top_n: int = 5) -> list[tuple[int, float]]:
    """对候选文本逐个打分并排序，返回 (原索引, 分数) 列表，按分数降序。

    Args:
        query: 用户问题（或改写后的问题）
        documents: 候选文本列表（通常是粗召回的 Top-20）
        top_n: 返回前几个

    Returns:
        [(原列表索引, relevance_score), ...]，长度 = min(top_n, len(documents))
    """
    settings = get_settings()
    if not documents:
        return []

    resp = httpx.post(
        f"{settings.siliconflow_base_url}/rerank",
        json={
            "model": RERANK_MODEL,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        },
        headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    results = resp.json()["results"]
    # API 已按分数降序返回；index 是原 documents 列表里的位置
    return [(item["index"], item["relevance_score"]) for item in results]
