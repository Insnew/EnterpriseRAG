"""问答接口：M4 起接入 LangGraph 管线，SSE 流式返回。

事件序列（前端按 event 类型分别处理）：
    event: meta      {"trace_id": "..."}     # 元信息（M7 可观测用）
    event: token     {"text": "旷"}          # 答案逐字
    event: citations {"citations": [...]}    # 引用来源
    event: done      {}                      # 正常结束
    event: error     {"message": "..."}      # 出错

协议细节：SSE = 服务器单向推送的 HTTP 长连接；响应头加
Cache-Control: no-cache 防止代理缓冲导致流式失效。
"""

import json
import logging
import uuid

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.api.schemas import ChatRequest
from app.rag.graph import build_graph

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chat"])


def _sse(event: str, data: dict) -> str:
    """按 SSE 协议打包一条事件：event 行 + data 行 + 空行分隔。"""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/api/v1/chat")
async def chat(req: ChatRequest) -> StreamingResponse:
    """流式问答：跑 LangGraph 管线，把 token/引用/结束事件逐个推给前端。"""
    trace_id = uuid.uuid4().hex[:12]
    history = [m.model_dump() for m in req.history]

    async def event_stream():
        yield _sse("meta", {"trace_id": trace_id})
        try:
            graph = build_graph()
            async for update in graph.astream(
                {"kb_id": req.kb_id, "question": req.question, "history": history},
                stream_mode="custom",  # 只接收节点内 writer() 推送的自定义事件
            ):
                yield _sse(update["event"], update)
        except Exception as exc:  # 异常也以 SSE 事件返回，前端能显示错误而非挂死
            logger.exception("问答失败")
            yield _sse("error", {"message": str(exc)})
        yield _sse("done", {})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
