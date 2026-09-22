"""评测统一入口。

用法：
    uv run python scripts/run_eval.py retrieval [--top-k 5] [--limit N]   # 检索指标（M2 起）
    uv run python scripts/run_eval.py ragas                                # 生成质量（M7 接入）
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging

setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="EnterpriseRAG 评测入口")
    sub = parser.add_subparsers(dest="command", required=True)

    p_retrieval = sub.add_parser("retrieval", help="检索质量评测（Recall@5/MRR）")
    p_retrieval.add_argument("--top-k", type=int, default=5)
    p_retrieval.add_argument("--limit", type=int, default=None)

    sub.add_parser("ragas", help="RAGAS 生成质量评测（M7 实现）")

    args = parser.parse_args()
    if args.command == "retrieval":
        from eval.retrieval_eval import run
        run(top_k=args.top_k, limit=args.limit)
    elif args.command == "ragas":
        parser.error("ragas 评测将在 M7 实现")


if __name__ == "__main__":
    main()
