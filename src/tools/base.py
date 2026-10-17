import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel

from src.core.config import get_settings
from src.core.exceptions import ToolError
from src.core.logging import get_logger

logger = get_logger(__name__)


class ToolResult(BaseModel):
    success: bool
    data: Any = None
    error: str = ""


@dataclass
class ToolDefinition:
    name: str
    description: str
    parameters: Dict[str, Any]  # JSON Schema
    execute: Callable
    category: str = "general"
    timeout: int = 30

    def to_openai_function(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    def __init__(self):
        self._tools: Dict[str, ToolDefinition] = {}
        self._settings = get_settings()

    def register(self, tool: ToolDefinition) -> None:
        self._tools[tool.name] = tool
        logger.info("Registered tool: %s [%s]", tool.name, tool.category)

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_all(self) -> List[ToolDefinition]:
        return list(self._tools.values())

    def list_by_category(self, category: str) -> List[ToolDefinition]:
        return [t for t in self._tools.values() if t.category == category]

    def to_openai_functions(self) -> List[Dict[str, Any]]:
        return [t.to_openai_function() for t in self._tools.values()]

    async def execute(self, name: str, arguments: Dict[str, Any]) -> str:
        tool = self._tools.get(name)
        if not tool:
            return json.dumps({"error": f"Tool '{name}' not found"}, ensure_ascii=False)

        start = time.monotonic()
        try:
            logger.info("Executing tool: %s", name, extra={"tool_args": str(arguments)[:200]})

            # Execute with timeout
            result = tool.execute(**arguments)
            if asyncio.iscoroutine(result):
                result = await asyncio.wait_for(result, timeout=tool.timeout)
            elif hasattr(result, "__await__"):
                result = await asyncio.wait_for(result, timeout=tool.timeout)

            elapsed = time.monotonic() - start
            logger.info("Tool %s completed in %.2fs", name, elapsed)

            return json.dumps(result, ensure_ascii=False, indent=2)

        except asyncio.TimeoutError:
            logger.error("Tool '%s' timed out after %ds", name, tool.timeout)
            raise ToolError(f"Tool '{name}' timed out", tool_name=name)
        except Exception as e:
            logger.error("Tool '%s' failed: %s", name, e)
            return json.dumps({"error": str(e)}, ensure_ascii=False)

    @property
    def tool_count(self) -> int:
        return len(self._tools)

    def get_categories(self) -> List[str]:
        return sorted(set(t.category for t in self._tools.values()))
