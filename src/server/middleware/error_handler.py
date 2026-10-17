from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.core.exceptions import AgentError, ConfigurationError, LLMError, SecurityError, ToolError
from src.core.logging import get_logger

logger = get_logger(__name__)


def setup_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(AgentError)
    async def agent_error_handler(request: Request, exc: AgentError):
        logger.error("Agent error: %s", exc.message)
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "type": type(exc).__name__, "detail": exc.detail},
        )

    @app.exception_handler(LLMError)
    async def llm_error_handler(request: Request, exc: LLMError):
        status = exc.status_code or 502
        logger.error("LLM error [%d]: %s", status, exc.message)
        return JSONResponse(
            status_code=status,
            content={"error": exc.message, "type": "LLMError"},
        )

    @app.exception_handler(SecurityError)
    async def security_error_handler(request: Request, exc: SecurityError):
        logger.warning("Security violation: %s", exc.message)
        return JSONResponse(
            status_code=403,
            content={"error": "Security violation", "type": "SecurityError"},
        )

    @app.exception_handler(ToolError)
    async def tool_error_handler(request: Request, exc: ToolError):
        logger.error("Tool error [%s]: %s", exc.tool_name, exc.message)
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "tool": exc.tool_name, "type": "ToolError"},
        )

    @app.exception_handler(ConfigurationError)
    async def config_error_handler(request: Request, exc: ConfigurationError):
        logger.error("Configuration error: %s", exc.message)
        return JSONResponse(
            status_code=500,
            content={"error": exc.message, "type": "ConfigurationError"},
        )

    @app.exception_handler(Exception)
    async def global_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled error: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error", "type": type(exc).__name__},
        )
