"""问答接口：M1 最简 LCEL 管道（非流式），M4 升级为 LangGraph 编排 + SSE 流式。

设计说明：prompt 要求 [1][2] 格式引用，M1 就先埋好引用溯源的结构，
避免后续大改——citations 与上下文顺序一一对应。
"""

from fastapi import APIRouter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.api.schemas import ChatRequest, ChatResponse, Citation
from app.llm.chat import get_chat_model
from app.storage.vector_store import search

router = APIRouter(tags=["chat"])

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名严谨的企业知识库助手。只依据提供的上下文回答问题，"
            "上下文不足以回答时明确说“知识库中未找到相关内容”。"
            "回答时用 [1][2] 格式标注引用来源。\n\n上下文：\n{context}",
        ),
        ("human", "{question}"),
    ]
)


def _format_docs(docs) -> str:
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1))


def _build_citations(docs) -> list[Citation]:
    return [
        Citation(
            index=i,
            file_name=doc.metadata.get("file_name", "unknown"),
            page=doc.metadata.get("page"),
            snippet=doc.page_content[:100],
        )
        for i, doc in enumerate(docs, start=1)
    ]


@router.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    docs = search(req.kb_id, req.question)
    chain = RAG_PROMPT | get_chat_model() | StrOutputParser()
    answer = await chain.ainvoke(
        {"context": _format_docs(docs), "question": req.question}
    )
    return ChatResponse(
        answer=answer,
        citations=_build_citations(docs),
        kb_id=req.kb_id,
    )
