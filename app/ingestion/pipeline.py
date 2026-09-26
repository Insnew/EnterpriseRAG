"""摄取编排（M6 版）：加载 → 元数据增强 → 父文档入库 → 切分 → 上下文生成 → 重建。

数据流（每个文件）：
    loader.load_file          → 页/篇级 Document
    metadata.enrich_metadata  → Markdown 节级切分 + updated_at
    docstore.save_parents     → 父文档入库，分配 parent_id
    splitter.split_documents  → 子 chunk（parent_id 自动继承；表格短路）
    contextual.add_context    → 每个子 chunk 生成前置上下文
最终：clear_kb + rebuild（Qdrant 与 docstore 同步全量重建）
"""

import asyncio
import logging
from pathlib import Path

from app.ingestion import contextual, loader, metadata, splitter
from app.storage import docstore
from app.storage.vector_store import rebuild

logger = logging.getLogger(__name__)


async def ingest_path(kb_id: str, path: Path) -> int:
    """摄入一个文件或目录，全量重建该知识库，返回写入的 chunk 总数。"""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())

    all_chunks = []
    for file in files:
        if file.suffix.lower() not in loader.SUPPORTED_SUFFIXES:
            logger.info("跳过不支持的文件: %s", file.name)
            continue
        try:
            docs = loader.load_file(file)
        except loader.UnsupportedFileTypeError as exc:
            logger.warning("加载失败 %s: %s", file.name, exc)
            continue

        docs = metadata.enrich_metadata(docs, file)          # 节级切分 + updated_at
        parents = docstore.save_parents(kb_id, docs)          # 父文档入库，得 parent_id
        chunks = splitter.split_documents(parents)            # parent_id 自动继承
        chunks = await contextual.add_context_to_chunks(chunks)  # 前置上下文
        for chunk in chunks:
            chunk.metadata["kb_id"] = kb_id
        all_chunks.extend(chunks)
        logger.info("已处理 %s：%d 个子 chunk", file.name, len(chunks))

    if not all_chunks:
        logger.warning("没有可摄入的内容，跳过重建")
        return 0

    # 全量重建：docstore 与 Qdrant 同步清理重灌
    docstore.clear_kb(kb_id)
    rebuild(kb_id, all_chunks)
    logger.info("摄入完成，共 %d 个子 chunk（全量重建）", len(all_chunks))
    return len(all_chunks)


def run_ingest(kb_id: str, path: Path) -> int:
    """同步入口（CLI 调用）：内部跑 asyncio。"""
    return asyncio.run(ingest_path(kb_id, path))
