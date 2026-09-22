"""检索质量评测：在固定评测集上跑当前检索管线，输出 Recall@5 / MRR / Hit@5 与分题型 breakdown。

评测粒度（M2 决策）：doc 级——检查返回的每个 chunk 的 file_name 是否命中
golden_doc_ids（任一命中即本题命中）。不经过 LLM，只调 BGE-M3 与向量库。

refusal 题（expect_refusal=true）在 M2 不参与计分：M1 管线没有相关性过滤，
拒答能力要等 M5 引入 judge 节点后才能评。此处仅计数。

用法：uv run python scripts/run_eval.py retrieval [--top-k 5] [--limit N]
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.storage.vector_store import search

DATASET = PROJECT_ROOT / "eval" / "dataset" / "questions.jsonl"
KB_ID = "eval"
DEFAULT_TOP_K = 5


def load_questions(limit: int | None = None) -> list[dict]:
    """读评测集，一行一条 JSON。limit 用于 CI 跑子集控制时长。"""
    lines = [
        json.loads(line)
        for line in DATASET.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines[:limit] if limit else lines


def score_one(item: dict, top_k: int) -> tuple[bool, float] | None:
    """跑一道题：检索 top_k 个 chunk，比对 file_name。

    返回 (hit, reciprocal_rank)；refusal 题返回 None（本阶段不评）。
    """
    if item["expect_refusal"]:
        return None
    docs = search(KB_ID, item["question"], k=top_k)
    names = [d.metadata.get("file_name", "") for d in docs]
    # 首个命中 golden 的位置（从 1 开始）；没命中 = 0
    first_pos = next(
        (i + 1 for i, name in enumerate(names) if name in item["golden_doc_ids"]),
        0,
    )
    hit = first_pos > 0
    rr = 1.0 / first_pos if hit else 0.0  # 倒数排名：第1位=1，第2位=0.5...
    return hit, rr


def summarize(results: list[tuple[str, bool, float]]) -> dict:
    """按题型汇总 Recall@5 / MRR。"""
    by_cat: dict[str, list[tuple[bool, float]]] = defaultdict(list)
    for category, hit, rr in results:
        by_cat[category].append((hit, rr))

    report = {}
    for category, scores in sorted(by_cat.items()):
        n = len(scores)
        recall = sum(1 for hit, _ in scores if hit) / n
        mrr = sum(rr for _, rr in scores) / n
        report[category] = {"n": n, "recall@5": round(recall, 3), "mrr": round(mrr, 3)}
    return report


def run(top_k: int = DEFAULT_TOP_K, limit: int | None = None) -> None:
    items = load_questions(limit)
    results: list[tuple[str, bool, float]] = []
    n_refusal = 0

    for item in items:
        outcome = score_one(item, top_k)
        if outcome is None:
            n_refusal += 1
            continue
        hit, rr = outcome
        results.append((item["category"], hit, rr))

    report = summarize(results)
    total = len(results)
    overall_recall = sum(1 for _, hit, _ in results if hit) / total
    overall_mrr = sum(rr for _, _, rr in results) / total

    # ===== 输出报告 =====
    print("\n========== 检索质量评测报告 ==========")
    print(f"评测集: {len(items)} 题 | 参与计分: {total} 题 | refusal 暂缓: {n_refusal} 题 | top_k={top_k}")
    print(f"{'题型':<12}{'题数':>6}{'Recall@5':>10}{'MRR':>10}")
    print("-" * 40)
    for category, m in sorted(report.items()):
        print(f"{category:<12}{m['n']:>6}{m['recall@5']:>10.3f}{m['mrr']:>10.3f}")
    print("-" * 40)
    print(f"{'总体':<12}{total:>6}{overall_recall:>10.3f}{overall_mrr:>10.3f}")
    print("======================================")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="检索质量评测")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--limit", type=int, default=None, help="只跑前 N 题（CI 用）")
    args = parser.parse_args()
    run(top_k=args.top_k, limit=args.limit)
