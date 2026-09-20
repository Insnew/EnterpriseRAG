"""统一日志配置：结构化输出，后续 M7 接入 trace_id 做全链路追踪。"""

import logging
import sys

CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    root = logging.getLogger()
    if root.handlers:  # uvicorn --reload 会重新 import，避免重复添加 handler
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt="%H:%M:%S"))
    root.addHandler(handler)
    root.setLevel(level)
    # 第三方库日志噪音大，统一压到 WARNING
    for noisy in ("httpx", "httpcore", "chromadb", "uvicorn.access"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
