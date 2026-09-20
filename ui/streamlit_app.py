"""演示界面：多轮对话 + 引用来源展示。

M1 为非流式请求；M4 切换为 SSE 流式（httpx stream + st.write_stream）。
启动：uv run streamlit run ui/streamlit_app.py
"""

import httpx
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(page_title="EnterpriseRAG 知识库问答", page_icon="📚")
st.title("📚 EnterpriseRAG 企业知识库问答")
st.caption("上传文档后用自然语言提问，答案附引用来源。")


def render_citations(citations: list[dict]) -> None:
    with st.expander("📎 引用来源"):
        for c in citations:
            page = f" · 第 {c['page']} 页" if c.get("page") is not None else ""
            st.markdown(f"**[{c['index']}] {c['file_name']}**{page}")
            st.markdown(f"> {c['snippet']}")


if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            render_citations(msg["citations"])

if question := st.chat_input("向知识库提问…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("检索中…"):
            try:
                resp = httpx.post(
                    f"{API_BASE}/api/v1/chat",
                    json={"question": question, "kb_id": "default", "history": []},
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                st.error(f"请求失败：{exc}。请确认后端已启动：uv run uvicorn app.main:app --reload")
                st.stop()
        st.markdown(data["answer"])
        if data.get("citations"):
            render_citations(data["citations"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": data["answer"],
            "citations": data.get("citations", []),
        }
    )
