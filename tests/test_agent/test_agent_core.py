"""AgentCore 单元测试：多轮消息拼装、最终答案、消息重置。"""
from types import SimpleNamespace

import pytest

from src.agent.core import AgentCore
from src.tools.base import ToolDefinition, ToolRegistry


class FakeLlm:
    def __init__(self, responses, max_iterations=5):
        self.responses = list(responses)
        self.settings = SimpleNamespace(agent_max_iterations=max_iterations)
        self.calls = []
        self.SYSTEM_PROMPT = "SYS_PROMPT"

    def build_system_message(self, rag_context: str = "") -> dict:
        return {"role": "system", "content": "SYS_PROMPT"}

    async def chat(self, messages, tools=None, stream=False):
        self.calls.append([dict(m) for m in messages])
        return self.responses.pop(0)


def tool_call(name, args, call_id="c1"):
    import json
    return {"id": call_id, "function": {"name": name, "arguments": json.dumps(args)}}


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="echo", description="", parameters={},
        execute=lambda msg="hi": {"echo": msg}, category="test",
    ))
    return reg


@pytest.fixture
def make_core(registry, monkeypatch):
    def _make(responses, max_iterations=5):
        import src.core.config as config_mod
        config_mod.get_settings().dashscope_api_key = "test-key"  # AgentCore 构造需要
        core = AgentCore(registry)
        core.llm = FakeLlm(responses, max_iterations)  # 替换为假客户端
        core.settings = core.llm.settings  # 让熔断轮数跟假 LLM 配置走，不改全局配置
        return core
    return _make


class TestRun:
    @pytest.mark.asyncio
    async def test_direct_answer(self, make_core):
        core = make_core([{"content": "答", "tool_calls": []}])
        result = await core.run("问题")
        assert result == {"answer": "答", "iterations": 1, "success": True}

    @pytest.mark.asyncio
    async def test_tool_round_trips_and_history_grows(self, make_core):
        core = make_core([
            {"content": "t", "tool_calls": [tool_call("echo", {"msg": "x"})]},
            {"content": "完成", "tool_calls": []},
        ])
        result = await core.run("问题")
        assert result["success"] is True
        assert result["iterations"] == 2
        # 消息历史: system, user, assistant(tool_calls), tool, assistant(最终)
        roles = [m["role"] for m in core.messages]
        assert roles == ["system", "user", "assistant", "tool", "assistant"]
        assert core.messages[3]["content"].find("x") != -1

    @pytest.mark.asyncio
    async def test_max_iterations_reported(self, make_core):
        core = make_core([
            {"content": "t", "tool_calls": [tool_call("echo", {})]},
        ] * 5, max_iterations=2)
        result = await core.run("问题")
        assert result["success"] is False
        assert result["iterations"] == 2

    def test_reset_clears_history(self, make_core):
        core = make_core([])
        core.messages = [{"role": "user", "content": "旧"}]
        core.reset()
        assert core.messages == []
