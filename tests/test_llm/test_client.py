"""LLMClient 单元测试：错误映射、重试行为、参数拼装。
绕过 __init__（不依赖真实 API key），直接注入 AsyncOpenAI mock。"""
import pytest
from unittest.mock import AsyncMock, MagicMock

from src.core.exceptions import LLMError, RateLimitError
from src.llm.client import LLMClient, SYSTEM_PROMPT


def make_client(create_mock: AsyncMock) -> LLMClient:
    client = LLMClient.__new__(LLMClient)  # 跳过 __init__，避免依赖 API key
    client.settings = MagicMock()
    client.settings.llm_model = "test-model"
    client.settings.llm_temperature = 0.2
    client.settings.llm_max_tokens = 256
    client._client = MagicMock()
    client._client.chat.completions.create = create_mock
    return client


class _Message:
    def model_dump(self):
        return {"role": "assistant", "content": "hello"}


class _Choice:
    def __init__(self):
        self.message = _Message()


class _Response:
    def __init__(self):
        self.choices = [_Choice()]


@pytest.mark.asyncio
async def test_chat_success_returns_message_dict():
    create = AsyncMock(return_value=_Response())
    client = make_client(create)
    result = await client.chat([{"role": "user", "content": "hi"}])
    assert result == {"role": "assistant", "content": "hello"}
    create.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_includes_tools_and_choice_auto():
    create = AsyncMock(return_value=_Response())
    client = make_client(create)
    tools = [{"type": "function", "function": {"name": "t"}}]
    await client.chat([{"role": "user", "content": "hi"}], tools=tools)
    kwargs = create.await_args.kwargs
    assert kwargs["tools"] == tools
    assert kwargs["tool_choice"] == "auto"
    assert kwargs["model"] == "test-model"
    assert kwargs["stream"] is False


@pytest.mark.asyncio
async def test_chat_without_tools_omits_tool_choice():
    create = AsyncMock(return_value=_Response())
    client = make_client(create)
    await client.chat([{"role": "user", "content": "hi"}])
    kwargs = create.await_args.kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_rate_limit_error_retried_three_times_then_raises():
    """429 错误触发指数退避：恰好尝试 3 次后 reraise。"""
    create = AsyncMock(side_effect=Exception("Error code: 429 - rate limited"))
    client = make_client(create)
    with pytest.raises(RateLimitError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert create.await_count == 3  # stop_after_attempt(3)


@pytest.mark.asyncio
async def test_generic_error_not_retried_raises_llm_error():
    """非限流错误不重试，直接映射为 LLMError。"""
    create = AsyncMock(side_effect=Exception("connection refused"))
    client = make_client(create)
    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert create.await_count == 1


def test_build_system_message_variants():
    client = make_client(AsyncMock())
    base = client.build_system_message()
    assert base["role"] == "system"
    assert base["content"] == SYSTEM_PROMPT

    with_ctx = client.build_system_message(rag_context="相关内容")
    assert "本地知识库相关内容" in with_ctx["content"]
    assert "相关内容" in with_ctx["content"]
