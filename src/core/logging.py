import logging
import sys
from typing import Any

_has_structlog = False
try:
    import structlog
    _has_structlog = True
except ImportError:
    structlog = None  # type: ignore


def configure_logging(level: str = "info") -> None:
    log_level = getattr(logging, level.upper(), logging.INFO)
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

    logging.basicConfig(format=fmt, stream=sys.stdout, level=log_level, datefmt="%Y-%m-%d %H:%M:%S")

    for name in ("uvicorn.access", "httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)

    if _has_structlog:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.stdlib.BoundLogger,
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )


def get_logger(name: str | None = None, **context: Any):
    if _has_structlog:
        logger = structlog.get_logger(name or __name__)
        return logger.bind(**context) if context else logger
    logger = logging.getLogger(name or __name__)
    return logger


def bind_context(logger, **context: Any):
    if _has_structlog:
        return logger.bind(**context)
    return logger
