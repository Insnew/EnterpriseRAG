"""CLI 摄入脚本。

用法：
    uv run python scripts/ingest.py default data/docs/
    uv run python scripts/ingest.py my_kb path/to/file.pdf
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.logging import setup_logging
from app.ingestion.pipeline import run_ingest

setup_logging()


def main() -> None:
    parser = argparse.ArgumentParser(description="摄入文档到知识库")
    parser.add_argument("kb_id", help="知识库 ID（如 default）")
    parser.add_argument("path", help="文件或目录路径")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        parser.error(f"路径不存在: {target}")
    run_ingest(args.kb_id, target)


if __name__ == "__main__":
    main()
