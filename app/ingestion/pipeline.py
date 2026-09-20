"""摄取编排：加载 → 切分 → 写入向量库。M6 在此插入 Contextual Chunking 与表格原子化。"""

import logging
from pathlib import Path

from app.ingestion import loader, splitter
from app.storage.vector_store import add_documents

logger = logging.getLogger(__name__)


def ingest_path(kb_id: str, path: Path) -> int:
    """摄入一个文件或目录，返回写入的 chunk 总数。"""
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    total = 0
    for file in files:
        if file.suffix.lower() not in loader.SUPPORTED_SUFFIXES:
            logger.info("跳过不支持的文件: %s", file.name)
            continue
        try:
            docs = loader.load_file(file)
        except loader.UnsupportedFileTypeError as exc:
            logger.warning("加载失败 %s: %s", file.name, exc)
            continue
        chunks = splitter.split_documents(docs)
        # kb_id 写进每个 chunk 的 metadata：collection 隔离之外的又一道过滤保障
        for chunk in chunks:
            chunk.metadata["kb_id"] = kb_id
        add_documents(kb_id, chunks)
        total += len(chunks)
        logger.info("已摄入 %s：%d 个 chunk", file.name, len(chunks))
    logger.info("摄入完成，共 %d 个 chunk", total)
    return total
