import json
from typing import Any, AsyncGenerator, List

from src.agent.types import AgentEvent, AgentResult, EventType, ToolCallEvent
from src.tools.base import ToolRegistry


class ReActEngine:
    """
    Re-Act engine: pure async generator that yields AgentEvent for each step.

    Usage:
        engine = ReActEngine(tool_registry, llm_client, rag_retriever)
        async for event in engine.execute("user query"):
            print(event)
    """

    def __init__(self, tool_registry: ToolRegistry, llm_client, rag_retriever=None):
        self.tool_registry = tool_registry
        self.llm = llm_client
        self.rag_retriever = rag_retriever

    async def execute(self, user_input: str, conversation_history: List[dict] | None = None) -> AgentResult:
        """Execute Re-Act loop and return the complete AgentResult."""
        events: List[AgentEvent] = []
        tool_calls_event: List[ToolCallEvent] = []

        tools = self.tool_registry.to_openai_functions()
        messages = conversation_history or []
        system_msg = self._build_system(user_input)
        messages = [system_msg, {"role": "user", "content": user_input}]

        for iteration in range(1, self.llm.settings.agent_max_iterations + 1):
            response = await self.llm.chat(messages, tools)
            msg_content = response.get("content") or ""
            llm_tool_calls = response.get("tool_calls", [])

            if msg_content:
                events.append(AgentEvent(
                    event_type=EventType.THOUGHT, iteration=iteration, content=msg_content
                ))

            messages.append({
                "role": "assistant", "content": msg_content,
                **({"tool_calls": llm_tool_calls} if llm_tool_calls else {}),
            })

            if not llm_tool_calls:
                return AgentResult(
                    success=True, answer=msg_content, events=events,
                    tool_calls=tool_calls_event, iterations=iteration,
                )

            for tc in llm_tool_calls:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                events.append(AgentEvent(
                    event_type=EventType.ACTION, iteration=iteration,
                    tool_name=func_name, tool_args=args,
                ))

                try:
                    result = await self.tool_registry.execute(func_name, args)
                except Exception as e:
                    result = json.dumps({"error": str(e)}, ensure_ascii=False)

                events.append(AgentEvent(
                    event_type=EventType.OBSERVATION, iteration=iteration,
                    tool_name=func_name, tool_result=result,
                ))

                tool_calls_event.append(ToolCallEvent(
                    id=tc.get("id", ""), name=func_name, arguments=args, result=result,
                ))

                messages.append({
                    "role": "tool", "tool_call_id": tc.get("id", ""), "content": result,
                })

        return AgentResult(
            success=False, answer="Reached max iterations.",
            events=events, tool_calls=tool_calls_event, iterations=iteration,
            error="max_iterations_exceeded",
        )

    async def stream(self, user_input: str) -> AsyncGenerator[AgentEvent, None]:
        """Execute Re-Act loop and stream AgentEvent as they happen."""
        tools = self.tool_registry.to_openai_functions()
        messages = [self._build_system(user_input), {"role": "user", "content": user_input}]

        for iteration in range(1, self.llm.settings.agent_max_iterations + 1):
            yield AgentEvent(event_type=EventType.ITERATION, iteration=iteration,
                             content=f"第 {iteration} 轮推理")

            response = await self.llm.chat(messages, tools)
            msg_content = response.get("content") or ""
            llm_tool_calls = response.get("tool_calls", [])

            if msg_content:
                yield AgentEvent(event_type=EventType.THOUGHT, iteration=iteration, content=msg_content)

            messages.append({
                "role": "assistant", "content": msg_content,
                **({"tool_calls": llm_tool_calls} if llm_tool_calls else {}),
            })

            if not llm_tool_calls:
                yield AgentEvent(event_type=EventType.FINAL_ANSWER, iteration=iteration, content=msg_content)
                return

            for tc in llm_tool_calls:
                func_name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"])

                yield AgentEvent(event_type=EventType.ACTION, iteration=iteration,
                                 tool_name=func_name, tool_args=args)

                try:
                    result = await self.tool_registry.execute(func_name, args)
                except Exception as e:
                    result = json.dumps({"error": str(e)}, ensure_ascii=False)

                yield AgentEvent(event_type=EventType.OBSERVATION, iteration=iteration,
                                 tool_name=func_name, tool_result=result)

                messages.append({
                    "role": "tool", "tool_call_id": tc.get("id", ""), "content": result,
                })

        yield AgentEvent(event_type=EventType.ERROR, iteration=iteration, content="Max iterations reached.")

    def _build_system(self, user_input: str = "") -> dict:
        msg = self.llm.build_system_message()
        if self.rag_retriever and self.rag_retriever.is_ready and user_input:
            context, _ = self.rag_retriever.query(user_input, top_k=2)
            if context:
                msg["content"] += f"\n\n## 本地知识库相关内容\n{context}"
        return msg
