"""文档加载器：M1 支持 PDF（数字版）与 Markdown/纯文本，M6 扩展 Word/Excel。

不用 langchain-community（官方已进入 sunset，无 1.0 正式版），直接基于
pymupdf / 标准库实现，代码量小且依赖可控。

扫描版 PDF（无文本层）属于已知范围外，检测到直接报错，暂不支持 OCR。
"""

import logging
from pathlib import Path

import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt"}


class UnsupportedFileTypeError(ValueError):
    """不支持的文件类型（含扫描版 PDF）。"""


def _load_pdf(path: Path) -> list[Document]:
    docs: list[Document] = []
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "file_name": path.name,
                        "page": page_index + 1,  # 页码从 1 开始，便于引用展示
                    },
                )
            )
    if not docs:
        raise UnsupportedFileTypeError(f"{path.name} 无文本层（可能是扫描版 PDF），暂不支持 OCR")
    return docs


def _load_text(path: Path) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path), "file_name": path.name, "page": None},
        )
    ]


def load_file(path: Path) -> list[Document]:
    """加载单个文件为 Document 列表，metadata 统一带 file_name 与 page。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(f"暂不支持的文件类型: {suffix}（支持 {SUPPORTED_SUFFIXES}）")

    if suffix == ".pdf":
        return _load_pdf(path)
    return _load_text(path)
