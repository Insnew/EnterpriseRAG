"""对话模型工厂：DeepSeek。

注意：deepseek-reasoner 不支持工具调用，后续 agent 阶段必须用 deepseek-chat。
"""

from langchain_deepseek import ChatDeepSeek

from app.core.config import get_settings


def get_chat_model() -> ChatDeepSeek:
    settings = get_settings()
    if not settings.deepseek_api_key:
        raise RuntimeError("缺少 DEEPSEEK_API_KEY，请先在 .env 中配置（参考 .env.example）")
    return ChatDeepSeek(
        model=settings.chat_model,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=0.3,
    )
