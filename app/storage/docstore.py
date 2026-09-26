"""父文档存储（ParentDocumentRetriever 的"父"层）。

父子检索两层结构：
- 子 chunk（400 字小块）进 Qdrant 参与检索——精准
- 父文档（节/页/整篇大块）存本 SQLite——完整

检索命中子 chunk 后，通过 metadata 里的 parent_id 回查父文档喂给 LLM。
"""

import json
import logging
import sqlite3
import uuid

from langchain_core.documents import Document

from app.core.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DB_PATH = PROJECT_ROOT / "data" / "docstore.db"


def _db() -> sqlite3.Connection:
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(_DB_PATH)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS parent_docs (
            parent_id TEXT PRIMARY KEY,
            kb_id TEXT NOT NULL,
            text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        )"""
    )
    return conn


def save_parents(kb_id: str, docs: list[Document]) -> list[Document]:
    """给每个大块 Document 分配 parent_id 并入库，返回带 parent_id 的 Document 列表。

    调用方拿返回值去切块——splitter 会把 parent_id 自动继承到每个子 chunk。
    """
    conn = _db()
    tagged: list[Document] = []
    for doc in docs:
        parent_id = uuid.uuid4().hex
        conn.execute(
            "INSERT OR REPLACE INTO parent_docs (parent_id, kb_id, text, metadata_json) "
            "VALUES (?, ?, ?, ?)",
            (parent_id, kb_id, doc.page_content, json.dumps(doc.metadata, ensure_ascii=False)),
        )
        new_meta = {**doc.metadata, "parent_id": parent_id}
        tagged.append(Document(page_content=doc.page_content, metadata=new_meta))
    conn.commit()
    logger.info("已存 %d 个父文档到 docstore", len(tagged))
    return tagged


def get_parents(kb_id: str, parent_ids: list[str]) -> list[Document]:
    """按 parent_id 批量取回父文档（保持传入顺序，去重）。"""
    conn = _db()
    out: list[Document] = []
    seen: set[str] = set()
    for pid in parent_ids:
        if pid in seen:
            continue
        seen.add(pid)
        row = conn.execute(
            "SELECT text, metadata_json FROM parent_docs WHERE parent_id = ? AND kb_id = ?",
            (pid, kb_id),
        ).fetchone()
        if row:
            out.append(Document(page_content=row[0], metadata=json.loads(row[1])))
    return out


def clear_kb(kb_id: str) -> None:
    """全量重建时清空该知识库的父文档（与 Qdrant collection 同步）。"""
    conn = _db()
    conn.execute("DELETE FROM parent_docs WHERE kb_id = ?", (kb_id,))
    conn.commit()
