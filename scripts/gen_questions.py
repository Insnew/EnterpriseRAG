"""批量生成评测题：读取 eval/dataset/docs/ 的文档，用 DeepSeek 按题型出题，写入 questions.jsonl。

用法：uv run python scripts/gen_questions.py

原理：把文档分批喂给 LLM，要求按固定 JSON 格式输出题目。
注意：LLM 生成内容不可全信——本脚本只做基础校验（字段完整/文件名合法），
golden_answer 与文档是否一致必须人工逐条复核（M2-3 核心环节）。
"""

import asyncio
import json
import logging
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging
from app.llm.chat import get_chat_model

setup_logging()
logger = logging.getLogger(__name__)

DOCS_DIR = PROJECT_ROOT / "eval" / "dataset" / "docs"
OUTPUT = PROJECT_ROOT / "eval" / "dataset" / "questions.jsonl"

# 批次设计：每批喂几份文档 + 本批要出的题型。
# cross_doc 单独一批——只喂互相引用的文档对，保证 LLM 有依据可查。
BATCHES = [
    {
        "docs": ["01-考勤管理制度.md", "02-薪酬管理制度.md", "03-差旅报销管理制度.md"],
        "categories": ["fact", "table", "multi_hop", "refusal"],
    },
    {
        "docs": ["04-招聘管理制度.md", "05-员工手册.md", "08-FAQ-新员工常见问题.md"],
        "categories": ["fact", "table", "refusal"],
    },
    {
        "docs": ["06-产品手册-星火OA系统.md", "07-产品手册-星火CRM系统.md"],
        "categories": ["fact", "table", "multi_hop"],
    },
    {
        "docs": ["09-培训管理制度.md", "10-信息安全管理制度.md", "11-绩效考核制度.md"],
        "categories": ["fact", "table", "multi_hop", "refusal"],
    },
    {
        "docs": ["12-办公设备管理制度.md", "13-印章证照管理制度.md", "14-会议管理制度.md", "15-突发事件应急预案.md"],
        "categories": ["fact", "table", "multi_hop"],
    },
    {
        # 跨文档题：文档间存在明确互相引用，答案要跨两份文档
        "docs": ["01-考勤管理制度.md", "02-薪酬管理制度.md", "06-产品手册-星火OA系统.md",
                 "07-产品手册-星火CRM系统.md", "10-信息安全管理制度.md", "11-绩效考核制度.md",
                 "15-突发事件应急预案.md"],
        "categories": ["cross_doc"],
    },
]

PROMPT_TEMPLATE = """你是企业知识库评测集设计专家。基于提供的公司内部文档，为 RAG 检索系统设计评测题。

本批要出的题型：{categories}

各题型定义：
- fact：事实题，答案在单一文档的某一段中明确写出
- table：表格数字题，答案在文档表格中（金额、比例、数量、配置等）
- multi_hop：多跳题，需结合同一文档的两个不同章节才能完整回答
- cross_doc：跨文档题，需结合两份不同文档才能回答（文档间有互相引用）
- refusal：拒答题，问的是文档明确不存在的服务或制度（如"公司食堂在哪"），
  golden_doc_ids 必须为空数组，golden_answer 填"知识库中无相关信息"

硬性要求：
1. 每类题型出 2-3 道，本批共 8-12 道
2. golden_answer 一句话精确作答，必须能在文档中找到依据，禁止编造
3. golden_doc_ids 只能使用我列出的文件名（一个或多个），严禁其他文件名
4. 问题口语化，像真实员工会问的样子
5. 只输出 JSON 数组，禁止任何解释文字或 markdown 围栏：
[{{"question": "...", "golden_answer": "...", "golden_doc_ids": ["文件名"], "category": "fact"}}]

本批可用文件名：{doc_names}

文档内容：
{docs_text}
"""


def load_docs(names: list[str]) -> str:
    parts = []
    for name in names:
        path = DOCS_DIR / name
        if not path.exists():
            logger.warning("文档不存在: %s", name)
            continue
        parts.append(f"===== {name} =====\n{path.read_text(encoding='utf-8')}")
    return "\n\n".join(parts)


def parse_json_array(text: str) -> list[dict]:
    """从 LLM 输出中抠出 JSON 数组（容忍 markdown 围栏与前后杂字）。"""
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError(f"未找到 JSON 数组: {text[:200]}")
    data = json.loads(match.group())
    if not isinstance(data, list):
        raise TypeError("输出不是数组")
    return data


def validate(item: dict, valid_names: set[str]) -> bool:
    """基础校验：字段完整、文件名合法、拒答题特殊规则。返回是否通过。"""
    required = {"question", "golden_answer", "golden_doc_ids", "category"}
    if not required.issubset(item):
        return False
    ids = item["golden_doc_ids"]
    if item["category"] == "refusal":
        if ids:  # 拒答题不允许有 golden 来源
            return False
    elif not ids or not set(ids).issubset(valid_names):
        return False
    return bool(item["question"].strip())


async def main() -> None:
    model = get_chat_model().bind(response_format={"type": "json_object"})
    all_items: list[dict] = []
    valid_names = {p.name for p in DOCS_DIR.glob("*.md")}

    for i, batch in enumerate(BATCHES, start=1):
        names = batch["docs"]
        prompt = PROMPT_TEMPLATE.format(
            categories="、".join(batch["categories"]),
            doc_names="、".join(names),
            docs_text=load_docs(names),
        )
        logger.info("批次 %d/%d：%s，生成中...", i, len(BATCHES), "、".join(batch["categories"]))
        resp = await model.ainvoke(prompt)
        items = parse_json_array(str(resp.content))
        accepted = [x for x in items if validate(x, valid_names)]
        logger.info("批次 %d：LLM 输出 %d 道，基础校验通过 %d 道", i, len(items), len(accepted))
        all_items.extend(accepted)

    # 统一编号 Q001... 后写入 jsonl
    with OUTPUT.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(all_items, start=1):
            record = {"id": f"Q{idx:03d}", **item, "expect_refusal": item["category"] == "refusal"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # 汇总
    from collections import Counter
    stats = Counter(x["category"] for x in all_items)
    logger.info("写入 %s：共 %d 道，分布 %s", OUTPUT, len(all_items), dict(stats))
    print("\n===== 初稿生成完毕 =====")
    print(f"共 {len(all_items)} 道题：{dict(stats)}")
    print("⚠️  下一步：人工逐条校验 golden_answer 与 golden_doc_ids（重点检查表格数字题！）")


if __name__ == "__main__":
    asyncio.run(main())
