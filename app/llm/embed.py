"""Embedding 模型工厂：BGE-M3 走硅基流动 OpenAI 兼容接口（免费额度）。

注意：更换 embedding 模型/服务后必须重建向量索引——
不同模型产出的向量空间不兼容，混存同一 collection 检索会失效。
"""

from langchain_openai import OpenAIEmbeddings

from app.core.config import get_settings


def get_embeddings() -> OpenAIEmbeddings:
    settings = get_settings()
    if not settings.siliconflow_api_key:
        raise RuntimeError("缺少 SILICONFLOW_API_KEY，请先在 .env 中配置（参考 .env.example）")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.siliconflow_api_key,
        base_url=settings.siliconflow_base_url,
        # bge-m3 不在 tiktoken 词表内，禁用 tiktoken 计数避免报错
        tiktoken_enabled=False,
    )
