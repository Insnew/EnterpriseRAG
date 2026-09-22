"""RAG 管线图：rewrite → retrieve → generate。

M4 版是直线图（为 LangGraph 模式打底）；M5 在 retrieve 后加入
rerank + judge 节点与条件边，才真正体现"图"的分支能力。

运行：graph = build_graph(); await graph.ainvoke({...})
"""

import logging

from langchain_core.output_parsers import StrOutputParser
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from app.llm.chat import get_chat_model
from app.rag.prompts import RAG_PROMPT, REWRITE_PROMPT
from app.rag.state import GraphState
from app.retrieval.rewrite import search_with_rewrite

logger = logging.getLogger(__name__)


def _format_docs(docs) -> str:
    """把检索结果拼成带 [1][2] 编号的上下文（原 chat.py 同名函数，迁移至此）。"""
    return "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, start=1))


def _build_citations(docs) -> list[dict]:
    return [
        {
            "index": i,
            "file_name": doc.metadata.get("file_name", "unknown"),
            "page": doc.metadata.get("page"),
            "snippet": doc.page_content[:100],
        }
        for i, doc in enumerate(docs, start=1)
    ]


# ---------- 节点 ----------

async def rewrite_node(state: GraphState) -> dict:
    """查询改写：结合最近对话历史，把指代问题补全成独立问题。

    无历史时直接跳过改写（省一次 LLM 调用、避免无谓改写）。
    """
    history = state.get("history") or []
    question = state["question"]
    if not history:
        logger.info("无对话历史，跳过改写")
        return {"rewritten_question": question}

    history_text = "\n".join(
        f"{msg['role']}: {msg['content']}" for msg in history[-6:]  # 最近 3 轮
    )
    prompt = REWRITE_PROMPT.format(history=history_text, question=question)
    rewritten = (await get_chat_model().ainvoke(prompt)).content.strip()
    logger.info("查询改写：%r -> %r", question, rewritten)
    return {"rewritten_question": rewritten}


async def retrieve_node(state: GraphState) -> dict:
    """混合检索：改写句 + 原句两路召回，RRF 融合。"""
    kb_id = state.get("kb_id", "default")
    docs = search_with_rewrite(
        rewritten=state["rewritten_question"],
        original=state["question"],
        kb_id=kb_id,
    )
    logger.info("两路召回融合后返回 %d 个 chunk", len(docs))
    return {"docs": docs}


async def generate_node(state: GraphState) -> dict:
    """生成答案：检索上下文 + 原始问题（注意：用原问题而非改写句）。

    流式输出：通过 get_stream_writer() 把每个 token 推给图外部的订阅者
    （chat 路由用 graph.astream(stream_mode="custom") 接收），
    前端就能实现打字机效果。
    """
    writer = get_stream_writer()
    docs = state["docs"]
    chain = RAG_PROMPT | get_chat_model() | StrOutputParser()

    parts: list[str] = []
    async for chunk in chain.astream(
        {"context": _format_docs(docs), "question": state["question"]}
    ):
        parts.append(chunk)
        writer({"event": "token", "text": chunk})

    answer = "".join(parts)
    writer({"event": "citations", "citations": _build_citations(docs)})
    return {"answer": answer, "citations": _build_citations(docs)}


# ---------- 建图 ----------

def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("rewrite", rewrite_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()
