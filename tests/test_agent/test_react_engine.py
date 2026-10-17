"""ReActEngine 单元测试：用脚本化假 LLM 驱动完整推理循环，覆盖
直接回答 / 工具调用 / 工具报错 / 超限熔断 / 流式事件序列 / RAG 注入。"""
import json
from types import SimpleNamespace

import pytest

from src.agent.react_engine import ReActEngine
from src.agent.types import EventType
from src.tools.base import ToolDefinition, ToolRegistry


class FakeLlm:
    """按脚本依次返回 chat 响应；记录每次收到的 messages 供断言。"""

    def __init__(self, responses, max_iterations=5):
        self.responses = list(responses)
        self.settings = SimpleNamespace(agent_max_iterations=max_iterations)
        self.calls = []

    def build_system_message(self, rag_context: str = "") -> dict:
        return {"role": "system", "content": "SYS_PROMPT"}

    async def chat(self, messages, tools=None, stream=False):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def tool_call(name: str, args: dict, call_id: str = "c1") -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="echo", description="回声工具", parameters={},
        execute=lambda msg="hi": {"echo": msg}, category="test",
    ))
    reg.register(ToolDefinition(
        name="boom", description="必定报错", parameters={},
        execute=lambda: (_ for _ in ()).throw(RuntimeError("炸了")), category="test",
    ))
    return reg


class TestExecute:
    @pytest.mark.asyncio
    async def test_direct_answer_without_tools(self, registry):
        llm = FakeLlm([{"content": "直接回答", "tool_calls": []}])
        engine = ReActEngine(registry, llm)
        result = await engine.execute("你好")
        assert result.success is True
        assert result.answer == "直接回答"
        assert result.iterations == 1
        assert result.tool_calls == []
        assert [e.event_type for e in result.events] == [EventType.THOUGHT]

    @pytest.mark.asyncio
    async def test_tool_call_then_final_answer(self, registry):
        llm = FakeLlm([
            {"content": "Thought: 调用 echo", "tool_calls": [tool_call("echo", {"msg": "hi"})]},
            {"content": "Final: 完成", "tool_calls": []},
        ])
        engine = ReActEngine(registry, llm)
        result = await engine.execute("测试")
        assert result.success is True
        assert result.iterations == 2
        assert result.answer == "Final: 完成"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "echo"
        assert json.loads(result.tool_calls[0].result)["echo"] == "hi"
        types = [e.event_type for e in result.events]
        assert types == [EventType.THOUGHT, EventType.ACTION, EventType.OBSERVATION, EventType.THOUGHT]

    @pytest.mark.asyncio
    async def test_tool_error_becomes_error_observation(self, registry):
        llm = FakeLlm([
            {"content": "t", "tool_calls": [tool_call("boom", {})]},
            {"content": "已捕获错误并作答", "tool_calls": []},
        ])
        engine = ReActEngine(registry, llm)
        result = await engine.execute("测试")
        assert result.success is True  # 工具报错不终止循环
        obs = [e for e in result.events if e.event_type == EventType.OBSERVATION][0]
        assert "炸了" in obs.tool_result
        # 报错结果作为 tool 角色回传给 LLM
        tool_msgs = [m for call in llm.calls for m in call if m.get("role") == "tool"]
        assert "炸了" in tool_msgs[0]["content"]

    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error_json(self, registry):
        llm = FakeLlm([
            {"content": "t", "tool_calls": [tool_call("nonexistent", {})]},
            {"content": "ok", "tool_calls": []},
        ])
        engine = ReActEngine(registry, llm)
        result = await engine.execute("测试")
        assert result.success is True
        assert "not found" in result.tool_calls[0].result

    @pytest.mark.asyncio
    async def test_max_iterations_fuse(self, registry):
        llm = FakeLlm([
            {"content": f"round {i}", "tool_calls": [tool_call("echo", {"msg": str(i)}, f"c{i}")]}
            for i in range(10)
        ], max_iterations=3)
        engine = ReActEngine(registry, llm)
        result = await engine.execute("一直调工具")
        assert result.success is False
        assert result.error == "max_iterations_exceeded"
        assert result.iterations == 3
        assert len(llm.responses) == 7  # 10 个脚本只消耗了 3 次配额

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_in_one_iteration(self, registry):
        llm = FakeLlm([
            {"content": "t", "tool_calls": [
                tool_call("echo", {"msg": "a"}, "c1"),
                tool_call("echo", {"msg": "b"}, "c2"),
            ]},
            {"content": "done", "tool_calls": []},
        ])
        engine = ReActEngine(registry, llm)
        result = await engine.execute("测试")
        assert len(result.tool_calls) == 2
        assert [tc.arguments["msg"] for tc in result.tool_calls] == ["a", "b"]


class TestStream:
    @pytest.mark.asyncio
    async def test_stream_event_sequence(self, registry):
        llm = FakeLlm([
            {"content": "t1", "tool_calls": [tool_call("echo", {"msg": "x"})]},
            {"content": "final", "tool_calls": []},
        ])
        engine = ReActEngine(registry, llm)
        events = [e async for e in engine.stream("测试")]
        types = [e.event_type for e in events]
        assert types == [
            EventType.ITERATION, EventType.THOUGHT,
            EventType.ACTION, EventType.OBSERVATION,
            EventType.ITERATION, EventType.THOUGHT, EventType.FINAL_ANSWER,
        ]
        assert events[-1].content == "final"

    @pytest.mark.asyncio
    async def test_stream_max_iterations_error_event(self, registry):
        llm = FakeLlm([
            {"content": "t", "tool_calls": [tool_call("echo", {})]},
        ] * 2, max_iterations=2)
        engine = ReActEngine(registry, llm)
        events = [e async for e in engine.stream("测试")]
        assert events[-1].event_type == EventType.ERROR


class TestRagInjection:
    class ReadyRetriever:
        is_ready = True

        def query(self, q, top_k=2):
            return "知识库片段内容", [{"source": "kb.md"}]

    @pytest.mark.asyncio
    async def test_rag_context_injected_into_system_message(self, registry):
        llm = FakeLlm([{"content": "ans", "tool_calls": []}])
        engine = ReActEngine(registry, llm, rag_retriever=self.ReadyRetriever())
        await engine.execute("什么是保修政策")
        system_msg = llm.calls[0][0]
        assert system_msg["role"] == "system"
        assert "本地知识库相关内容" in system_msg["content"]
        assert "知识库片段内容" in system_msg["content"]

    @pytest.mark.asyncio
    async def test_no_rag_when_retriever_absent(self, registry):
        llm = FakeLlm([{"content": "ans", "tool_calls": []}])
        engine = ReActEngine(registry, llm)
        await engine.execute("问题")
        assert "本地知识库" not in llm.calls[0][0]["content"]
