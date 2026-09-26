"""metadata 增强测试：Markdown 节级切分 + updated_at + 切分后继承。"""

from pathlib import Path

from langchain_core.documents import Document

from app.ingestion.metadata import enrich_metadata
from app.ingestion.splitter import split_documents

MD_TEXT = (
    "# 考勤管理制度\n\n"
    "## 迟到规定\n第三条 迟到扣款 20 元。\n\n"
    "## 请假规定\n第十五条 病假规则。\n"
)


def test_markdown_split_by_sections(tmp_path: Path):
    path = tmp_path / "考勤.md"
    path.write_text(MD_TEXT, encoding="utf-8")
    docs = [Document(page_content=MD_TEXT, metadata={"file_name": path.name})]

    enriched = enrich_metadata(docs, path)
    sections = [d.metadata.get("section") for d in enriched]
    # 一级标题 "# 考勤管理制度" 自成首节，随后两个 ## 各成一节
    assert len(enriched) == 3, f"应切出 3 节，实际 {len(enriched)}"
    assert "迟到规定" in sections and "请假规定" in sections
    assert all(d.metadata.get("updated_at") for d in enriched), "应带 updated_at"
    # 节内容应保留标题行（语境完整）
    assert any("## 迟到规定" in d.page_content for d in enriched)


def test_markdown_without_headings_single_section(tmp_path: Path):
    path = tmp_path / "plain.md"
    path.write_text("没有标题的纯文本内容。", encoding="utf-8")
    docs = [Document(page_content="没有标题的纯文本内容。", metadata={"file_name": path.name})]

    enriched = enrich_metadata(docs, path)
    assert len(enriched) == 1
    assert enriched[0].metadata["section"] == ""


def test_non_markdown_passthrough(tmp_path: Path):
    """PDF/Excel 等非 Markdown 只补 updated_at，不做节级切分。"""
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 dummy")
    docs = [Document(page_content="PDF 内容", metadata={"file_name": path.name})]

    enriched = enrich_metadata(docs, path)
    assert len(enriched) == 1
    assert enriched[0].metadata.get("updated_at")
    assert "section" not in enriched[0].metadata


def test_section_inherited_after_split(tmp_path: Path):
    """节级 Document 切块后，section 应自动继承到子 chunk。"""
    path = tmp_path / "长文.md"
    long_section = "## 第五章\n" + "第五条 内容说明。" * 100  # 超过 400 字强制切块
    path.write_text(long_section, encoding="utf-8")
    docs = [Document(page_content=long_section, metadata={"file_name": path.name})]

    enriched = enrich_metadata(docs, path)
    chunks = split_documents(enriched)
    assert len(chunks) > 1, "长节应被切出多个 chunk"
    assert all(c.metadata.get("section") == "第五章" for c in chunks), "section 应继承到所有子 chunk"
