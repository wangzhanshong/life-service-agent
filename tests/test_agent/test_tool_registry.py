import pytest

from src.tools.base import ToolDefinition, ToolRegistry


class TestToolRegistry:
    def test_register_and_get(self):
        reg = ToolRegistry()
        tool = ToolDefinition(name="test", description="A test tool", parameters={}, execute=lambda: "ok")
        reg.register(tool)
        assert reg.tool_count == 1
        assert reg.get("test") is tool
        assert reg.get("nonexistent") is None

    def test_register_duplicate(self):
        reg = ToolRegistry()
        t1 = ToolDefinition(name="t", description="1", parameters={}, execute=lambda: 1)
        t2 = ToolDefinition(name="t", description="2", parameters={}, execute=lambda: 2)
        reg.register(t1)
        reg.register(t2)
        assert reg.tool_count == 1

    def test_categories(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="a", description="", parameters={}, execute=lambda: 1, category="x"))
        reg.register(ToolDefinition(name="b", description="", parameters={}, execute=lambda: 1, category="y"))
        assert reg.get_categories() == ["x", "y"]
        assert len(reg.list_by_category("x")) == 1

    def test_to_openai_functions(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="calc", description="Calculate", category="compute",
            parameters={"type": "object", "properties": {"expr": {"type": "string"}}, "required": ["expr"]},
            execute=lambda expr: expr,
        ))
        funcs = reg.to_openai_functions()
        assert len(funcs) == 1
        assert funcs[0]["type"] == "function"
        assert funcs[0]["function"]["name"] == "calc"

    @pytest.mark.asyncio
    async def test_execute_success(self):
        reg = ToolRegistry()
        reg.register(ToolDefinition(name="echo", description="", parameters={}, execute=lambda msg="hi": {"echo": msg}))
        result = await reg.execute("echo", {"msg": "hello"})
        assert "hello" in result

    @pytest.mark.asyncio
    async def test_execute_not_found(self):
        reg = ToolRegistry()
        result = await reg.execute("nonexistent", {})
        assert "not found" in result
