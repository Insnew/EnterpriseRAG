"""RAGAS 生成质量评测：端到端跑 LangGraph 图产出（问题, 答案, 检索上下文），
用 RAGAS 四指标（LLM 判卷）打分。

与检索评测（retrieval_eval.py）的分工：
- 检索评测：只调 search()，评"找没找对"（Recall@5/MRR，机器批卷）
- 本评测：跑完整图（rewrite→retrieve→rerank→judge→generate），
  评"答没答好"（RAGAS，LLM 判卷）

四指标：
- Faithfulness（忠实度）：答案每句话都能在检索上下文找到依据（防幻觉）
- AnswerRelevancy（答案相关性）：答案是否紧扣问题
- SemanticSimilarity（语义相似度）：答案与标准答案的语义接近程度
- ContextPrecisionWithReference（上下文精确度）：检索上下文的贡献密度

成本控制：54 题取 30 题（按题型分层抽样、确定性可复现），refusal 题不参与
（拒答准确率由检索评测单独统计）。judge 用 deepseek-chat——自己评自己存在
共模偏差，实习项目可接受，面试主动说明是加分。

用法：uv run python eval/ragas_eval.py [--full] [--limit N]
"""

import asyncio
import json
import sys
import types
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# ragas 0.4.3 硬 import 了 langchain-community 已 sunset 移除的
# chat_models.vertexai 模块（仅用于判断 LLM 是否支持 multi-completion，
# 本项目 judge 用 DeepSeek 不涉及）。塞占位类绕过；ragas 发新版修复后删除。
_placeholder = types.ModuleType("langchain_community.chat_models.vertexai")
_placeholder.ChatVertexAI = type("ChatVertexAI", (), {})  # type: ignore[assignment]
sys.modules["langchain_community.chat_models.vertexai"] = _placeholder

from openai import AsyncOpenAI, OpenAI
from ragas.embeddings.openai_provider import OpenAIEmbeddings as RagasEmbeddings
from ragas.llms import llm_factory
from ragas.metrics.collections import (
    AnswerRelevancy,
    ContextPrecisionWithReference,
    Faithfulness,
    SemanticSimilarity,
)

from app.core.config import get_settings
from app.rag.graph import build_graph

DATASET = PROJECT_ROOT / "eval" / "dataset" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
KB_ID = "eval"
SAMPLE_SIZE = 30  # 默认子集：54 题 → 抽 30（refusal 题除外）


def load_questions() -> list[dict]:
    """读评测集，一行一条 JSON。"""
    return [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sample_questions(items: list[dict], size: int) -> list[dict]:
    """按题型分层抽样：每类按占比取前 N 题（jsonl 原始顺序，确定性可复现）。"""
    eligible = [i for i in items if not i["expect_refusal"]]
    by_cat: dict[str, list[dict]] = {}
    for item in eligible:
        by_cat.setdefault(item["category"], []).append(item)

    picked: list[dict] = []
    for group in by_cat.values():
        n = max(1, round(len(group) / len(eligible) * size))
        picked.extend(group[:n])
    return picked


async def _run_one(item: dict) -> dict:
    """跑完整图，产出 RAGAS 需要的四件套。

    contexts 用 reranked_docs（重排后实际喂给 LLM 的父文档）——
    Faithfulness 评的是"答案忠于实际看到的上下文"，不是粗召回。
    """
    result = await build_graph().ainvoke({"kb_id": KB_ID, "question": item["question"]})
    return {
        "user_input": item["question"],
        "response": result["answer"],
        "retrieved_contexts": [d.page_content for d in result["reranked_docs"]],
        "reference": item["golden_answer"],
        "category": item["category"],
    }


async def produce_rows(items: list[dict]) -> list[dict]:
    """串行跑图（避免撞 API 限流），每题约 10-15s，30 题约 5-8 分钟。"""
    rows = []
    for i, item in enumerate(items, start=1):
        print(f"[{i}/{len(items)}] 跑图: {item['id']} {item['question'][:30]}...")
        rows.append(await _run_one(item))
    return rows


def build_judge() -> tuple:
    """judge LLM（deepseek-chat）+ BGE-M3 嵌入（硅基流动 OpenAI 兼容）。

    collections 指标只收 modern embeddings（embedding_factory 产物），
    不接受 langchain 包装器，因此直接包硅基流动的 OpenAI 兼容 client。

    注意需要两个 embeddings 实例：
    - AnswerRelevancy 走 aembed_text（async 路径）→ 要 async client
    - SemanticSimilarity 只走 embed_text（sync 路径）→ 若给 async client，
      ragas 0.4.3 会在运行中的事件循环里 run_until_complete 死锁 → 要 sync client
    """
    settings = get_settings()
    llm_client = AsyncOpenAI(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
    )
    # max_tokens 拉满：Faithfulness/ContextPrecision 需对 5 个父文档（可上万字）
    # 输出长 JSON，DeepSeek 默认上限会截断导致 IncompleteOutputException
    judge = llm_factory(
        settings.chat_model, provider="openai", client=llm_client, max_tokens=8192
    )

    embed_kwargs = {
        "api_key": settings.siliconflow_api_key,
        "base_url": settings.siliconflow_base_url,
    }
    emb_async = RagasEmbeddings(
        client=AsyncOpenAI(**embed_kwargs), model=settings.embedding_model
    )
    emb_sync = RagasEmbeddings(
        client=OpenAI(**embed_kwargs), model=settings.embedding_model
    )
    return judge, emb_async, emb_sync


async def main(full: bool = False, limit: int | None = None) -> None:
    items = load_questions()
    if limit:
        picked = items[:limit]
    elif full:
        picked = [i for i in items if not i["expect_refusal"]]  # 全量 50 题
    else:
        picked = sample_questions(items, SAMPLE_SIZE)
    print(f"评测 {len(picked)} 题（refusal 题除外）")

    # ① 跑图产出答案与上下文（串行）
    rows = await produce_rows(picked)

    # ② RAGAS 判卷
    # 注意：不经过 evaluate()——ragas 0.4.3 的 collections 新指标
    # （SimpleBaseMetric 体系）过不了老 evaluate() 的 isinstance(m, Metric)
    # 检查，直接逐样本并发调 ascore()（指标内部自动并发多步 LLM 调用）。
    judge, emb_async, emb_sync = build_judge()
    metrics = [
        Faithfulness(llm=judge),
        AnswerRelevancy(llm=judge, embeddings=emb_async),
        SemanticSimilarity(embeddings=emb_sync),
        ContextPrecisionWithReference(llm=judge),
    ]

    sem = asyncio.Semaphore(4)  # 样本级并发 4，避免撞 DeepSeek/硅基流动限流

    # 各指标所需输入不同（如 SemanticSimilarity 不认 user_input），按名映射
    _ASCORE_ARGS = {
        "faithfulness": ("user_input", "response", "retrieved_contexts"),
        "answer_relevancy": ("user_input", "response"),
        "semantic_similarity": ("reference", "response"),
        "context_precision_with_reference": ("user_input", "reference", "retrieved_contexts"),
    }

    async def _score_one(row: dict) -> dict:
        async with sem:
            scores = {"category": row["category"]}
            for m in metrics:
                kwargs = {k: row[k] for k in _ASCORE_ARGS[m.name]}
                res = await m.ascore(**kwargs)
                scores[m.name] = res.value
            return scores

    per_row = await asyncio.gather(*[_score_one(r) for r in rows])

    # ③ 汇总（总体 + 分题型）与落盘
    names = [m.name for m in metrics]

    def _mean(items: list[dict]) -> dict:
        return {name: round(sum(x[name] for x in items) / len(items), 3) for name in names}

    summary = _mean(per_row)
    by_cat: dict[str, list[dict]] = {}
    for x in per_row:
        by_cat.setdefault(x["category"], []).append(x)

    print("\n========== RAGAS 生成质量评测报告 ==========")
    print(f"题数: {len(rows)} | judge: deepseek-chat（自评，存在共模偏差）")
    header = f"{'指标':<28}" + "".join(f"{n.split('_')[0][:12]:>14}" for n in names)
    print(header)
    print("-" * (28 + 14 * len(names)))
    for category, items in sorted(by_cat.items()):
        m = _mean(items)
        print(f"{category:<28}" + "".join(f"{m[n]:>14.3f}" for n in names))
    print("-" * (28 + 14 * len(names)))
    print(f"{'总体':<28}" + "".join(f"{summary[n]:>14.3f}" for n in names))
    print("============================================")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {
        "date": datetime.now(tz=UTC).strftime("%Y-%m-%d"),
        "n": len(rows),
        "metrics": summary,
        "by_category": {c: _mean(items) for c, items in sorted(by_cat.items())},
    }
    out_path = RESULTS_DIR / "ragas_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"结果已保存: {out_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS 生成质量评测")
    parser.add_argument("--full", action="store_true", help="跑全部非 refusal 题（50 题）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试用）")
    args = parser.parse_args()
    asyncio.run(main(full=args.full, limit=args.limit))
