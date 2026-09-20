"""API 请求/响应模型。"""

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    kb_id: str = Field("default", max_length=64)
    history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class Citation(BaseModel):
    index: int
    file_name: str
    page: int | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    kb_id: str


class HealthResponse(BaseModel):
    status: str
    version: str
