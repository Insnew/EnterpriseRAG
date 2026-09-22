"""LangGraph 图状态：所有节点共享的"工作台"。

每个节点（rewrite/retrieve/generate）从 state 里读输入，把结果写回 state。
M4 只用前几个字段；judged/refusal 等字段 M5 引入。
"""

from typing import NotRequired, TypedDict

from langchain_core.documents import Document


class GraphState(TypedDict):
    kb_id: str                          # 知识库 ID（检索节点用它定位 collection）
    question: str                       # 用户原始问题（生成答案时用它，而非改写句）
    history: NotRequired[list[dict]]    # 最近几轮对话 [{"role":..., "content":...}]
    rewritten_question: NotRequired[str]  # 改写后的独立问题（检索用它）
    docs: NotRequired[list[Document]]   # 混合检索结果
    answer: NotRequired[str]            # 最终答案
    citations: NotRequired[list[dict]]  # 引用来源（file_name/page/snippet）
