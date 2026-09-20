"""M1 冒烟测试：全部离线，不调用任何真实 API。

覆盖：模块装配、模型工厂参数、文档加载（MD/PDF）、切分、向量库构建、/health。
"""

from pathlib import Path

import pymupdf
from fastapi.testclient import TestClient
from langchain_deepseek import ChatDeepSeek
from langchain_openai import OpenAIEmbeddings

from app.core.config import Settings
from app.ingestion import splitter
from app.ingestion.loader import load_file
from app.storage.vector_store import get_vector_store

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_DOC = PROJECT_ROOT / "samples" / "示例-星火科技考勤管理制度.md"


def test_app_imports():
    import app.main  # noqa: F401  导入即验证路由与依赖装配无误


def test_settings_defaults():
    s = Settings(_env_file=None)
    assert s.chat_model == "deepseek-chat"
    assert s.embedding_model == "BAAI/bge-m3"
    assert s.retrieval_top_k == 4


def test_chat_model_factory():
    m = ChatDeepSeek(
        model="deepseek-chat",
        api_key="sk-dummy",
        base_url="https://api.deepseek.com",
        temperature=0.3,
    )
    assert m.model_name == "deepseek-chat"


def test_embedding_factory():
    e = OpenAIEmbeddings(
        model="BAAI/bge-m3",
        api_key="sk-dummy",
        base_url="https://api.siliconflow.cn/v1",
        tiktoken_enabled=False,
    )
    assert e.model == "BAAI/bge-m3"


def test_load_markdown():
    docs = load_file(SAMPLE_DOC)
    assert docs, "样本文档应能加载"
    assert all(d.metadata["file_name"] == SAMPLE_DOC.name for d in docs)


def test_load_pdf(tmp_path):
    """用 pymupdf 生成最小 PDF 验证提取链路（含中文，依赖系统中文字体）。"""
    pdf_path = tmp_path / "test.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    cjk_font = Path(r"C:\Windows\Fonts\msyh.ttc")
    if cjk_font.exists():
        page.insert_font(fontname="cjk", fontfile=str(cjk_font))
        page.insert_text((72, 72), "星火科技考勤制度：旷工扣款为当日工资的200%。", fontname="cjk")
    else:
        page.insert_text((72, 72), "SparkTech attendance policy.")
    doc.save(pdf_path)
    doc.close()

    docs = load_file(pdf_path)
    text = "".join(d.page_content for d in docs)
    assert "星火科技" in text or "SparkTech" in text
    assert docs[0].metadata["page"] == 1


def test_splitter_on_sample():
    docs = load_file(SAMPLE_DOC)
    chunks = splitter.split_documents(docs)
    assert len(chunks) >= 3, "400 字切分下样本应产出多个 chunk"
    assert all(len(c.page_content) <= 500 for c in chunks)


def test_vector_store_constructs(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "sk-dummy")
    monkeypatch.setenv("CHROMA_DIR", str(tmp_path))
    monkeypatch.setenv("ANONYMIZED_TELEMETRY", "False")
    store = get_vector_store("test_kb")
    assert store._collection.name == "kb_test_kb"


def test_health_endpoint():
    from app.main import app

    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": "0.1.0"}
