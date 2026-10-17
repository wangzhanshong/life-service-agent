class AgentError(Exception):
    """Base exception for all agent-related errors."""
    def __init__(self, message: str, detail: str = ""):
        self.message = message
        self.detail = detail
        super().__init__(message)


class ConfigurationError(AgentError):
    """Raised when configuration is invalid or missing."""
    pass


class LLMError(AgentError):
    """Raised when LLM API call fails."""
    def __init__(self, message: str, status_code: int | None = None, detail: str = ""):
        self.status_code = status_code
        super().__init__(message, detail)


class ToolError(AgentError):
    """Raised when a tool execution fails."""
    def __init__(self, message: str, tool_name: str = "", detail: str = ""):
        self.tool_name = tool_name
        super().__init__(message, detail)


class SecurityError(AgentError):
    """Raised when a security violation is detected (e.g., path traversal)."""
    pass


class RAGError(AgentError):
    """Raised when RAG operations fail."""
    pass


class ModelNotFoundError(LLMError):
    """Raised when requested model is not available."""
    pass


class RateLimitError(LLMError):
    """Raised when API rate limit is exceeded."""
    pass
