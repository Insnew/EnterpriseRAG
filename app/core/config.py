"""全局配置：pydantic-settings 读取 .env，字段与 .env.example 一一对应。"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """应用配置。缺失必填项时在启动阶段直接报错，避免运行时才发现。"""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM（DeepSeek）
    deepseek_api_key: str = ""
    chat_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    # Embedding（硅基流动，OpenAI 兼容接口）
    siliconflow_api_key: str = ""
    siliconflow_base_url: str = "https://api.siliconflow.cn/v1"
    embedding_model: str = "BAAI/bge-m3"

    # 存储
    chroma_dir: str = "./data/chroma"

    # 检索
    retrieval_top_k: int = 4


@lru_cache
def get_settings() -> Settings:
    """进程内单例，避免每次调用都重读 .env。"""
    return Settings()
