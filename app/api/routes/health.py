"""健康检查：M7 扩展为含依赖探测（向量库/DB）的完整探针。"""

from fastapi import APIRouter

from app.api.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.1.0")
