from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx
from openai import AsyncOpenAI
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.core.config import get_settings
from src.core.exceptions import ConfigurationError, LLMError, RateLimitError
from src.core.logging import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """你是一个生活服务综合智能体（AI Agent），采用 Re-Act（推理-行动）模式工作。
你能够使用多种工具来帮助用户完成复杂任务。

## 能力范围
1. **RAG 知识检索**: 从本地知识库检索信息
2. **文件系统操作**: 在工作目录下读取、写入、搜索、浏览文件
3. **数学与数据工具**: 安全计算数学表达式、处理日期、统计文本
4. **MCP 远程服务**: 天气查询、地理信息、MBTI测试、美食推荐

## 工作原则
- 仔细分析用户需求，确定需要使用哪些工具
- 每次只调用一个工具，观察结果后再决定下一步
- 如果任务简单可以直接回答，无需使用工具
- 最终给出清晰、完整的回答
- 使用中文回复用户"""


class LLMClient:
    """Async OpenAI-compatible LLM client with retry logic and timeout."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings

        if not settings.is_llm_configured:
            raise ConfigurationError(
                "LLM API key not configured. Set DASHSCOPE_API_KEY environment variable."
            )

        timeout = httpx.Timeout(settings.llm_timeout, connect=10.0)
        self._client = AsyncOpenAI(
            api_key=settings.dashscope_api_key,
            base_url=settings.llm_api_base,
            timeout=timeout,
            max_retries=0,  # We handle retries ourselves via tenacity
        )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((RateLimitError,)),
        reraise=True,
    )
    async def chat(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """Send a chat completion request to the LLM."""
        kwargs = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": stream,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            logger.debug("LLM request", model=self.settings.llm_model, msg_count=len(messages))
            response = await self._client.chat.completions.create(**kwargs)
            return response.choices[0].message.model_dump()
        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "429" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {error_str}")
            raise LLMError(f"LLM API call failed: {error_str}") from e

    async def chat_stream(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Stream chat completion responses."""
        kwargs = {
            "model": self.settings.llm_model,
            "messages": messages,
            "temperature": self.settings.llm_temperature,
            "max_tokens": self.settings.llm_max_tokens,
            "stream": True,
        }

        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        try:
            stream = await self._client.chat.completions.create(**kwargs)
            async for chunk in stream:
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta:
                    yield delta.model_dump()
        except Exception as e:
            error_str = str(e)
            if "rate" in error_str.lower() or "429" in error_str:
                raise RateLimitError(f"Rate limit exceeded: {error_str}")
            raise LLMError(f"LLM stream failed: {error_str}") from e

    def build_system_message(self, rag_context: str = "") -> Dict[str, str]:
        """Build system message with optional RAG context."""
        content = SYSTEM_PROMPT
        if rag_context:
            content += f"\n\n## 本地知识库相关内容\n{rag_context}"
        return {"role": "system", "content": content}

    @property
    def is_available(self) -> bool:
        return self.settings.is_llm_configured


# Singleton instance
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the global LLM client instance."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
