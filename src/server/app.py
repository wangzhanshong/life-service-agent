from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.core.config import get_settings
from src.core.logging import configure_logging, get_logger
from src.models.base import create_tables
from src.server.middleware.error_handler import setup_error_handlers
from src.server.middleware.request_id import RequestIDMiddleware
from src.server.middleware.timing import TimingMiddleware
from src.server.routers import chat, health, tools

logger = get_logger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging(settings.server_log_level)
    logger.info("Starting Life Service Agent server...")

    # Initialize database
    await create_tables()
    logger.info("Database tables created")

    # Initialize RAG
    from src.server.dependencies import init_rag
    init_rag()
    logger.info("RAG system initialized")

    yield

    logger.info("Server shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="生活服务综合智能体 (Life Service AI Agent)",
        description="基于大模型的Re-Act模式AI Agent，集成RAG检索增强、文件操作、计算工具及MCP服务",
        version="2.0.0",
        lifespan=lifespan,
    )

    # Middleware
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(TimingMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins.split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Error handlers
    setup_error_handlers(app)

    # Routers
    app.include_router(health.router, tags=["health"])
    app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
    app.include_router(tools.router, prefix="/api/v1", tags=["tools"])

    # Static files (Web UI)
    static_dir = Path(__file__).parent / "static"
    static_dir.mkdir(exist_ok=True)
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")

    return app
