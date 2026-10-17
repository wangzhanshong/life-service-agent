import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from src.core.logging import get_logger

logger = get_logger(__name__)


class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.monotonic()
        response: Response = await call_next(request)
        elapsed = (time.monotonic() - start) * 1000
        response.headers["X-Response-Time-ms"] = f"{elapsed:.2f}"
        logger.info(
            "%s %s - %d (%.2fms)",
            request.method, request.url.path,
            response.status_code, elapsed,
        )
        return response
