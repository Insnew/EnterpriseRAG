"""向量库工厂：M3 起切换到 Qdrant（Docker 独立服务，生产标配）。

与 M1 Chroma 版的关键差异：
- named vectors：每条记录带 dense（BGE-M3 语义）+ sparse（中文 BM25 关键词）双向量
- HYBRID 检索：Qdrant 服务端 Query API 两路召回 + RRF 融合，返回单一排序结果
- BM25 词表按知识库持久化到 data/bm25/{kb_id}.json——入库与查询必须同一份词表

M3 简化决策：ingest 即全量重建（先 fit 新词表 → 删旧 collection → 重写）。
增量更新在 M7 引入（词表与索引版本化）。search() 接口保持不变，调用方零改动。
"""

import logging

from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.llm.embed import get_embeddings
from app.retrieval.sparse_bm25 import ChineseBM25SparseEmbeddings, sparse_model_path

logger = logging.getLogger(__name__)

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    """Qdrant 客户端单例：连接 localhost:6333 的 Docker 服务。"""
    global _client
    if _client is None:
        _client = QdrantClient(url=get_settings().qdrant_url)
    return _client


def _load_sparse(kb_id: str) -> ChineseBM25SparseEmbeddings:
    path = sparse_model_path(kb_id)
    if not path.exists():
        raise RuntimeError(
            f"知识库 {kb_id} 的 BM25 词表不存在（{path}），请先运行 ingest 摄入文档"
        )
    return ChineseBM25SparseEmbeddings.load(path)


def get_vector_store(kb_id: str) -> QdrantVectorStore:
    """HYBRID 检索的向量库（查询/写入共用）。"""
    return QdrantVectorStore(
        client=_get_client(),
        collection_name=f"kb_{kb_id}",
        embedding=get_embeddings(),                    # dense：BGE-M3
        sparse_embedding=_load_sparse(kb_id),          # sparse：中文 BM25
        retrieval_mode=RetrievalMode.HYBRID,           # 两路召回 + 服务端 RRF
        vector_name="dense",
        sparse_vector_name="sparse",
        # 查询路径 collection 必然已存在（ingest 建好），跳过配置校验提速
        validate_collection_config=False,
    )


def rebuild(kb_id: str, docs: list[Document]) -> int:
    """全量重建一个知识库：fit 词表 → 删旧 collection → 写入全部 chunk。"""
    client = _get_client()
    collection = f"kb_{kb_id}"

    # 1. 用全量语料训练 BM25 词表并持久化（查询时加载同一份）
    sparse = ChineseBM25SparseEmbeddings.fit([d.page_content for d in docs])
    sparse.save(sparse_model_path(kb_id))
    logger.info("BM25 词表已更新：%d 个词，avgdl=%.1f", len(sparse.vocab), sparse.avgdl)

    # 2. 删除旧 collection（全量重建语义；不存在时忽略 404）
    try:
        client.delete_collection(collection)
        logger.info("已删除旧 collection: %s", collection)
    except UnexpectedResponse:
        logger.info("旧 collection %s 不存在，跳过删除", collection)

    # 3. 显式创建 collection（langchain-qdrant 1.1.0 的 add_texts 不会自动建库）
    client.create_collection(
        collection_name=collection,
        vectors_config={
            "dense": qdrant_models.VectorParams(
                size=1024, distance=qdrant_models.Distance.COSINE
            ),
        },
        sparse_vectors_config={
            "sparse": qdrant_models.SparseVectorParams(),
        },
    )
    logger.info("已创建 collection: %s（dense=1024维 COSINE + sparse）", collection)

    # 4. 写入全部 chunk（add_documents 内部自动为每条记录算 dense + sparse 双向量）
    store = QdrantVectorStore(
        client=client,
        collection_name=collection,
        embedding=get_embeddings(),
        sparse_embedding=sparse,
        retrieval_mode=RetrievalMode.HYBRID,
        vector_name="dense",
        sparse_vector_name="sparse",
        # collection 刚被删除，构造时自动新建；跳过配置校验
        validate_collection_config=False,
    )
    store.add_documents(docs)
    logger.info("已写入 %d 条记录到 %s", len(docs), collection)
    return len(docs)


def search(kb_id: str, query: str, k: int | None = None) -> list[Document]:
    """混合检索：dense + sparse 两路召回经 RRF 融合，返回 Top-K。"""
    top_k = k or get_settings().retrieval_top_k
    return get_vector_store(kb_id).similarity_search(query, k=top_k)
