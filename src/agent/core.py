import json
from typing import Any, AsyncGenerator, Dict, List, Optional

from src.core.config import get_settings
from src.core.exceptions import AgentError
from src.core.logging import get_logger
from src.llm.client import get_llm_client
from src.tools.base import ToolRegistry

logger = get_logger(__name__)


class AgentCore:
    """Async Agent core with LLM interaction and tool dispatch."""

    def __init__(self, tool_registry: ToolRegistry, rag_retriever=None):
        self.tool_registry = tool_registry
        self.rag_retriever = rag_retriever
        self.settings = get_settings()
        self.llm = get_llm_client()
        self.messages: List[Dict[str, Any]] = []

    def _build_system_message(self, user_input: str = "") -> Dict[str, str]:
        content = self.llm.SYSTEM_PROMPT  # type: ignore[attr-defined]
        if self.rag_retriever and self.rag_retriever.is_ready and user_input:
            context, _ = self.rag_retriever.query(user_input, top_k=2)
            if context:
                content += f"\n\n## 本地知识库相关内容\n{context}"
        return {"role": "system", "content": content}

    async def run(self, user_input: str) -> Dict[str, Any]:
        """Run agent and return the final result."""
        tools = self.tool_registry.to_openai_functions()
        system_msg = self._build_system_message(user_input)
        self.messages = [system_msg, {"role": "user", "content": user_input}]

        for iteration in range(1, self.settings.agent_max_iterations + 1):
            response = await self.llm.chat(self.messages, tools)
            msg_content = response.get("content") or ""
            tool_calls = response.get("tool_calls", [])

            assistant_msg: Dict = {"role": "assistant", "content": msg_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if not tool_calls:
                return {"answer": msg_content, "iterations": iteration, "success": True}

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])
                try:
                    result = await self.tool_registry.execute(func_name, arguments)
                except Exception as e:
                    result = json.dumps({"error": str(e)}, ensure_ascii=False)

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

        return {"answer": "Reached maximum iterations without a final answer.", "iterations": iteration, "success": False}

    async def stream(self, user_input: str) -> AsyncGenerator[Dict[str, Any], None]:
        """Run agent and stream each step as it happens."""
        from src.agent.types import EventType

        tools = self.tool_registry.to_openai_functions()
        system_msg = self._build_system_message(user_input)
        self.messages = [system_msg, {"role": "user", "content": user_input}]

        for iteration in range(1, self.settings.agent_max_iterations + 1):
            yield {"event_type": EventType.ITERATION, "iteration": iteration, "content": f"第 {iteration} 轮推理"}

            response = await self.llm.chat(self.messages, tools)
            msg_content = response.get("content") or ""
            tool_calls = response.get("tool_calls", [])

            if msg_content:
                yield {"event_type": EventType.THOUGHT, "iteration": iteration, "content": msg_content}

            assistant_msg: Dict = {"role": "assistant", "content": msg_content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            self.messages.append(assistant_msg)

            if not tool_calls:
                yield {"event_type": EventType.FINAL_ANSWER, "iteration": iteration, "content": msg_content}
                return

            for tc in tool_calls:
                func_name = tc["function"]["name"]
                arguments = json.loads(tc["function"]["arguments"])

                yield {
                    "event_type": EventType.ACTION,
                    "iteration": iteration,
                    "tool_name": func_name,
                    "tool_args": arguments,
                }

                try:
                    result = await self.tool_registry.execute(func_name, arguments)
                except Exception as e:
                    result = json.dumps({"error": str(e)}, ensure_ascii=False)

                yield {
                    "event_type": EventType.OBSERVATION,
                    "iteration": iteration,
                    "tool_name": func_name,
                    "tool_result": result,
                }

                self.messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", ""),
                    "content": result,
                })

        yield {"event_type": EventType.ERROR, "iteration": iteration, "content": "Reached maximum iterations."}

    def reset(self) -> None:
        self.messages = []
