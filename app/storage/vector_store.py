"""向量库工厂：M1 用 Chroma 本地持久化（零部署依赖）。

每个知识库一个 collection（kb_{kb_id}），kb_id 同时写入 chunk metadata——
多知识库隔离从 M1 就打底。M3 切换 Qdrant（Chroma 不支持稀疏向量，混合检索需要），
届时只替换本文件实现，调用方无感知。
"""

from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.core.config import get_settings
from app.llm.embed import get_embeddings


def get_vector_store(kb_id: str) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=f"kb_{kb_id}",
        persist_directory=settings.chroma_dir,
        embedding_function=get_embeddings(),
        collection_metadata={"hnsw:space": "cosine"},
    )


def add_documents(kb_id: str, docs: list[Document]) -> None:
    get_vector_store(kb_id).add_documents(docs)


def search(kb_id: str, query: str, k: int | None = None) -> list[Document]:
    top_k = k or get_settings().retrieval_top_k
    return get_vector_store(kb_id).similarity_search(query, k=top_k)
