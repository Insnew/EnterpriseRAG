"""Contextual Chunking：入库前给每个 chunk 生成一句前置上下文。

解决的问题：切分后的 chunk 是"孤儿"（如"第三条：扣款20元"脱离了文件标题），
向量化时语义不完整。用 LLM 结合元数据生成一句说明拼在 chunk 前面：

    "这是《考勤管理制度》第五章中关于迟到扣款标准的条款。\n第三条：扣款20元"

成本控制（每个 chunk 一次 LLM 调用 = 花钱）：
- SQLite 缓存（data/context_cache.db）：key = 内容 hash，内容没变的 chunk
  重跑 ingest 时不重复调用、不重复花钱
- 表格 chunk 跳过：表头自含语境
- asyncio.gather 并发调用
"""

import asyncio
import hashlib
import logging
import sqlite3

from langchain_core.documents import Document

from app.core.config import PROJECT_ROOT
from app.llm.chat import get_chat_model

logger = logging.getLogger(__name__)

_CACHE_PATH = PROJECT_ROOT / "data" / "context_cache.db"

CONTEXT_PROMPT = """你是企业文档的语境标注员。请为下面的文本片段生成一句简短的前置说明，
说明这段文本的语境（它属于哪份文档、哪个章节、讲的是什么），便于脱离原文也能被理解。

要求：
1. 只输出一句说明，不超过 30 个字，不要任何标点包裹或解释
2. 结合给出的文档信息，不要凭空编造

文档信息：文件「{file_name}」{section_info}

文本片段：
{chunk_text}

前置说明："""


def _db() -> sqlite3.Connection:
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_CACHE_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS context_cache (content_hash TEXT PRIMARY KEY, prefix TEXT)"
    )
    return conn


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def _prefix_for(doc: Document) -> str:
    """为单个 chunk 生成前置说明（先查缓存，未命中再调 LLM 并写缓存）。"""
    key = _content_hash(doc.page_content)
    conn = _db()
    row = conn.execute("SELECT prefix FROM context_cache WHERE content_hash = ?", (key,)).fetchone()
    if row:
        return row[0]

    section = doc.metadata.get("section")
    section_info = f"「{section}」章节" if section else "未提供章节信息"
    prompt = CONTEXT_PROMPT.format(
        file_name=doc.metadata.get("file_name", "未知文件"),
        section_info=section_info,
        chunk_text=doc.page_content[:800],
    )
    prefix = (await get_chat_model().ainvoke(prompt)).content.strip()
    conn.execute(
        "INSERT OR REPLACE INTO context_cache (content_hash, prefix) VALUES (?, ?)",
        (key, prefix),
    )
    conn.commit()
    return prefix


async def add_context_to_chunks(chunks: list[Document]) -> list[Document]:
    """给普通文本 chunk 生成前置上下文并拼入 page_content；表格 chunk 原样跳过。"""
    targets = [c for c in chunks if c.metadata.get("chunk_kind") != "table"]
    tables = [c for c in chunks if c.metadata.get("chunk_kind") == "table"]

    if not targets:
        return chunks

    logger.info("为 %d 个 chunk 生成前置上下文（%d 个表格跳过）", len(targets), len(tables))
    prefixes = await asyncio.gather(*[_prefix_for(doc) for doc in targets])

    enriched: list[Document] = []
    for doc, prefix in zip(targets, prefixes, strict=True):
        enriched.append(
            Document(
                page_content=f"{prefix}\n{doc.page_content}",
                metadata={**doc.metadata, "context_prefix": prefix},
            )
        )
    logger.info("上下文生成完成（缓存命中 + 新生成）")
    return enriched + tables
