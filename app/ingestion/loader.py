"""文档加载器：支持 PDF（数字版）/ Markdown / 文本 / Word / Excel。

M6 新增：
- Word（python-docx 逐段读）
- Excel（openpyxl 每 sheet 转 Markdown 表格，chunk_kind="table"）
- PDF 表格检测（pdfplumber 提取表格转 Markdown 作为额外 table chunk）

表格原子化：chunk_kind="table" 的 Document 会被 splitter 短路（永不切开），
保证整张表作为一个检索单元。

已知权衡：PDF 的表格文字在正文 chunk 里也会出现一次（正文照旧全文提取），
少量重叠换取"表格完整性 + 正文完整性"双保证，注释留档。
扫描版 PDF（无文本层）仍明确不支持 OCR。
"""

import logging
from pathlib import Path

import docx
import openpyxl
import pdfplumber
import pymupdf
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".pdf", ".md", ".txt", ".docx", ".xlsx"}


class UnsupportedFileTypeError(ValueError):
    """不支持的文件类型（含扫描版 PDF）。"""


def _table_to_markdown(table: list[list]) -> str:
    """二维数组转 Markdown 表格文本（openpyxl/pdfplumber 共用）。"""
    if not table or not any(row for row in table if any(cell for cell in row)):
        return ""
    rows = []
    for row in table:
        cells = [str(cell).strip().replace("|", "\\|") if cell is not None else "" for cell in row]
        rows.append("| " + " | ".join(cells) + " |")
    # 表头与首行之间插入分隔行
    header_sep = "| " + " | ".join(["---"] * len(table[0])) + " |"
    return "\n".join([rows[0], header_sep, *rows[1:]])


# ---------- PDF ----------

def _load_pdf(path: Path) -> list[Document]:
    docs: list[Document] = []
    # 正文：pymupdf 逐页取文本（保证不丢字）
    with pymupdf.open(path) as pdf:
        for page_index, page in enumerate(pdf):
            text = page.get_text("text").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path),
                            "file_name": path.name,
                            "page": page_index + 1,
                        },
                    )
                )
    if not docs:
        raise UnsupportedFileTypeError(f"{path.name} 无文本层（可能是扫描版 PDF），暂不支持 OCR")

    # 表格：pdfplumber 检测每页表格 → 额外 table chunk（保证表格完整可查）
    with pdfplumber.open(path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            for table in page.extract_tables():
                md = _table_to_markdown(table)
                if md:
                    docs.append(
                        Document(
                            page_content=md,
                            metadata={
                                "source": str(path),
                                "file_name": path.name,
                                "page": page_index + 1,
                                "chunk_kind": "table",  # splitter 看到此标记不切分
                            },
                        )
                    )
    return docs


# ---------- Word ----------

def _load_docx(path: Path) -> list[Document]:
    d = docx.Document(str(path))
    # 段落 + 表格文字都收进来，保持顺序；交给 splitter 切块
    parts: list[str] = []
    for para in d.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())
    for table in d.tables:
        md = _table_to_markdown([[cell.text for cell in row.cells] for row in table.rows])
        if md:
            parts.append(md)
    if not parts:
        raise UnsupportedFileTypeError(f"{path.name} 内容为空")
    return [
        Document(
            page_content="\n\n".join(parts),
            metadata={"source": str(path), "file_name": path.name, "page": None},
        )
    ]


# ---------- Excel ----------

def _load_xlsx(path: Path) -> list[Document]:
    wb = openpyxl.load_workbook(str(path), data_only=True)
    docs: list[Document] = []
    for sheet in wb.worksheets:
        rows = [[cell for cell in row] for row in sheet.iter_rows(values_only=True)]
        md = _table_to_markdown(rows)
        if not md:
            continue
        docs.append(
            Document(
                page_content=md,
                metadata={
                    "source": str(path),
                    "file_name": path.name,
                    "page": None,
                    "sheet_name": sheet.title,
                    "chunk_kind": "table",  # Excel 整表作为一个检索单元
                },
            )
        )
    if not docs:
        raise UnsupportedFileTypeError(f"{path.name} 无有效内容")
    return docs


# ---------- 文本 ----------

def _load_text(path: Path) -> list[Document]:
    return [
        Document(
            page_content=path.read_text(encoding="utf-8"),
            metadata={"source": str(path), "file_name": path.name, "page": None},
        )
    ]


# ---------- 入口 ----------

_LOADERS = {
    ".pdf": _load_pdf,
    ".docx": _load_docx,
    ".xlsx": _load_xlsx,
}


def load_file(path: Path) -> list[Document]:
    """加载单个文件为 Document 列表，metadata 统一带 file_name 与 page。"""
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise UnsupportedFileTypeError(f"暂不支持的文件类型: {suffix}（支持 {SUPPORTED_SUFFIXES}）")

    if suffix in _LOADERS:
        return _LOADERS[suffix](path)
    return _load_text(path)
