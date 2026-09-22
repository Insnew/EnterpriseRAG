"""摄取编排：加载 → 切分 → 全量重建向量库（M3 起 Qdrant 混合检索）。

M3 简化决策：每次 ingest 视为全量重建——BM25 词表需要在全量语料上训练，
先收集全部 chunk 再一次性 rebuild，词表与索引始终一致。
增量更新在 M7 引入。
"""

import logging
from pathlib import Path

from app.ingestion import loader, splitter
from app.storage.vector_store import rebuild

logger = logging.getLogger(__name__)


def ingest_path(kb_id: str, path: Path) -> int:
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
        chunks = splitter.split_documents(docs)
        # kb_id 写进每个 chunk 的 metadata：collection 隔离之外的又一道过滤保障
        for chunk in chunks:
            chunk.metadata["kb_id"] = kb_id
        all_chunks.extend(chunks)
        logger.info("已切分 %s：%d 个 chunk", file.name, len(chunks))

    if not all_chunks:
        logger.warning("没有可摄入的内容，跳过重建")
        return 0

    rebuild(kb_id, all_chunks)
    logger.info("摄入完成，共 %d 个 chunk（全量重建）", len(all_chunks))
    return len(all_chunks)
