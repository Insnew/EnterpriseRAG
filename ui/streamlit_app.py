"""演示界面：SSE 流式多轮对话 + 引用来源展示。

M4：改为 httpx.stream 接收 SSE（打字机效果），客户端维护最近几轮
对话历史传给后端做查询改写。

启动：uv run streamlit run ui/streamlit_app.py
"""

import json

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"
MAX_HISTORY = 6  # 最近 3 轮（6 条消息）传给后端做指代消解

st.set_page_config(page_title="EnterpriseRAG 知识库问答", page_icon="📚")
st.title("📚 EnterpriseRAG 企业知识库问答")
st.caption("支持多轮追问（如\"那迟到呢？\"），答案附引用来源，逐字流式输出。")


def render_citations(citations: list[dict]) -> None:
    with st.expander("📎 引用来源"):
        for c in citations:
            page = f" · 第 {c['page']} 页" if c.get("page") is not None else ""
            st.markdown(f"**[{c['index']}] {c['file_name']}**{page}")
            st.markdown(f"> {c['snippet']}")


def parse_sse_line(line: str) -> tuple[str | None, dict | None]:
    """解析一行 SSE：event 行记录类型，data 行解析 JSON。"""
    if line.startswith("event: "):
        return line[7:], None
    if line.startswith("data: "):
        return None, json.loads(line[6:])
    return None, None


if "messages" not in st.session_state:
    st.session_state.messages = []

# 渲染历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

if question := st.chat_input("向知识库提问…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # 组装最近 3 轮历史（不含刚提问的这一条）
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ][-MAX_HISTORY:]

    answer_box = st.chat_message("assistant")
    placeholder = answer_box.empty()
    buffer = ""
    citations = []

    try:
        with httpx.stream(
            "POST",
            f"{API_BASE}/api/v1/chat",
            json={"question": question, "kb_id": "default", "history": history},
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            current_event = None
            for line in resp.iter_lines():
                event, data = parse_sse_line(line)
                if event:
                    current_event = event
                if data is not None:
                    if current_event == "token":
                        buffer += data["text"]
                        placeholder.markdown(buffer)
                    elif current_event == "citations":
                        citations = data["citations"]
                    elif current_event == "error":
                        placeholder.error(f"请求失败：{data['message']}")
    except httpx.HTTPError as exc:
        placeholder.error(f"请求失败：{exc}。请确认后端已启动：uv run uvicorn app.main:app --reload")
        st.stop()

    st.session_state.messages.append(
        {"role": "assistant", "content": buffer, "citations": citations}
    )
    if citations:
        with answer_box:
            render_citations(citations)
