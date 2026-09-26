"""chunk 元数据增强：Markdown 按标题切节 + 文档更新时间。

Markdown 按 #/## 标题切成节级 Document（每节一个），section 记入 metadata——
splitter 切块时 metadata 自动继承到每个子 chunk，引用展示与
Contextual Chunking（步5）都能用到。节级 Document 同时是步6 父子检索的父文档候选。

非 Markdown（PDF/Word/Excel）不做节级切分：PDF 有 page 元数据、
Excel 整表即单元、Word 的 heading 解析留待后续（M6 范围控制）。
"""

import re
from datetime import UTC, datetime
from pathlib import Path

from langchain_core.documents import Document

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$")


def _split_markdown_sections(text: str, base_meta: dict) -> list[Document]:
    """按 #/## 标题把 Markdown 文本切成节级 Document。

    每个节的内容包含其标题行（保留语境）；无标题的文本整篇作为一个节。
    """
    lines = text.splitlines()
    docs: list[Document] = []
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        m = _HEADING_RE.match(line)
        if m:
            # 遇到新标题：先归档上一节
            if current_lines:
                docs.append(
                    Document(
                        page_content="\n".join(current_lines),
                        metadata={**base_meta, "section": current_title},
                    )
                )
            current_title = m.group(2)
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:  # 归档最后一节
        docs.append(
            Document(
                page_content="\n".join(current_lines),
                metadata={**base_meta, "section": current_title},
            )
        )
    return docs


def enrich_metadata(docs: list[Document], path: Path) -> list[Document]:
    """入口：所有类型补 updated_at；Markdown 额外做节级切分。"""
    updated = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).strftime("%Y-%m-%d")
    out: list[Document] = []
    for doc in docs:
        doc.metadata["updated_at"] = updated
        is_md = path.suffix.lower() == ".md"
        is_table = doc.metadata.get("chunk_kind") == "table"
        if is_md and not is_table:
            out.extend(_split_markdown_sections(doc.page_content, dict(doc.metadata)))
        else:
            out.append(doc)
    return out
