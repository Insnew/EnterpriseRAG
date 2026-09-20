"""FastAPI 入口：uv run uvicorn app.main:app --reload"""

from fastapi import FastAPI

from app.api.routes import chat, health
from app.core.logging import setup_logging

setup_logging()

app = FastAPI(title="EnterpriseRAG", version="0.1.0")
app.include_router(health.router)
app.include_router(chat.router)


@app.get("/")
async def root() -> dict:
    return {"name": "EnterpriseRAG", "docs": "/docs"}
