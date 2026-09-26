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
from app.llm.rerank import rerank
from app.rag.prompts import JUDGE_PROMPT, RAG_PROMPT, REWRITE_PROMPT
from app.rag.state import GraphState
from app.retrieval.parent_retriever import retrieve_parents
from app.retrieval.rewrite import search_with_rewrite

# 粗召回宽度与重排截断数（两阶段检索：宽进窄出）
CANDIDATE_K = 20
RERANK_TOP_K = 5

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
    """粗召回：改写句 + 原句两路召回，RRF 融合，放宽到 Top-20。

    M5 起是两阶段检索的第一阶段（粗排）：宁可多捞不漏，
    排序质量交给 rerank 节点修准。

    M6 起召回后做父子替换：子 chunk 命中 → 换回父文档（完整段落），
    下游 rerank/judge/generate 读到的是完整上下文而非碎片。
    """
    kb_id = state.get("kb_id", "default")
    child_docs = search_with_rewrite(
        rewritten=state["rewritten_question"],
        original=state["question"],
        kb_id=kb_id,
        top_k=CANDIDATE_K,
    )
    docs = retrieve_parents(kb_id, child_docs, top_n=CANDIDATE_K)
    logger.info("粗召回 %d 个子 chunk，父文档替换后 %d 个候选", len(child_docs), len(docs))
    return {"docs": docs}


async def rerank_node(state: GraphState) -> dict:
    """精排：用 bge-reranker 对候选逐对打分，按分数重排后截断 Top-K。

    注意：rerank 分数只用于排序，不用于"相关/不相关"阈值判断
    （分数跨查询不可比）——拒答判断由 M5 的 judge 节点负责。
    """
    docs = state["docs"]
    if not docs:
        return {"reranked_docs": []}
    results = rerank(
        query=state["rewritten_question"],
        documents=[d.page_content for d in docs],
        top_n=RERANK_TOP_K,
    )
    reranked = [docs[idx] for idx, _ in results]
    logger.info("重排后截断为 %d 个 chunk", len(reranked))
    return {"reranked_docs": reranked}


async def judge_node(state: GraphState) -> dict:
    """相关性闸门：LLM 判断重排后的内容能否回答用户问题。

    为什么不用 rerank 分数设阈值：分数跨查询不可比（尺度会浮动），
    LLM 语义判断更稳。这是"低相关拒答"防幻觉的关键一环。
    """
    docs = state["reranked_docs"]
    prompt = JUDGE_PROMPT.format(
        question=state["rewritten_question"],
        context=_format_docs(docs),
    )
    result = (await get_chat_model().ainvoke(prompt)).content.strip().lower()
    judge = "yes" if "yes" in result else "no"  # 容错：容忍 LLM 输出 "yes。" 等变体
    logger.info("相关性判断：%s", judge)
    return {"judge_result": judge}


async def refuse_node(state: GraphState) -> dict:
    """拒答：不生成答案，直接返回固定话术 + 空引用（防幻觉）。"""
    writer = get_stream_writer()
    answer = "知识库中未找到相关内容。"
    writer({"event": "token", "text": answer})
    writer({"event": "citations", "citations": []})
    return {"answer": answer, "citations": []}


async def generate_node(state: GraphState) -> dict:
    """生成答案：检索上下文 + 原始问题（注意：用原问题而非改写句）。

    流式输出：通过 get_stream_writer() 把每个 token 推给图外部的订阅者
    （chat 路由用 graph.astream(stream_mode="custom") 接收），
    前端就能实现打字机效果。
    """
    writer = get_stream_writer()
    docs = state["reranked_docs"]  # M5：生成用重排后的结果
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
    graph.add_node("rerank", rerank_node)
    graph.add_node("judge", judge_node)
    graph.add_node("generate", generate_node)
    graph.add_node("refuse", refuse_node)

    graph.add_edge(START, "rewrite")
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "judge")
    # 条件边：图的第一次分叉——相关走生成，不相关走拒答
    graph.add_conditional_edges(
        "judge",
        lambda state: "generate" if state.get("judge_result") == "yes" else "refuse",
        {"generate": "generate", "refuse": "refuse"},
    )
    graph.add_edge("generate", END)
    graph.add_edge("refuse", END)

    return graph.compile()
