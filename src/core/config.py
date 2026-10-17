import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Base directory
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent.parent)

    # LLM
    dashscope_api_key: str = Field(default="", alias="DASHSCOPE_API_KEY")
    llm_model: str = Field(default="qwen-plus")
    llm_api_base: str = Field(default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    llm_max_tokens: int = Field(default=2048)
    llm_temperature: float = Field(default=0.7)
    llm_timeout: int = Field(default=60, description="LLM API timeout in seconds")
    llm_max_retries: int = Field(default=3)

    # Embedding
    embedding_model: str = Field(default="text-embedding-v2")
    embedding_dimension: int = Field(default=1536, description="Fallback dimension when API unavailable")

    # Agent
    agent_max_iterations: int = Field(default=10)
    agent_workspace_dir: str = Field(default="./workspace")
    agent_tool_timeout: int = Field(default=30, description="Tool execution timeout in seconds")

    # RAG
    rag_chunk_size: int = Field(default=500)
    rag_chunk_overlap: int = Field(default=50)
    rag_top_k: int = Field(default=3)
    rag_index_path: str = Field(default="./data/faiss_index")
    rag_docs_path: str = Field(default="./data/documents")

    # Database
    database_url: str = Field(default="sqlite+aiosqlite:///./data/agent.db")

    # Server
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    server_log_level: str = Field(default="info")
    cors_origins: str = Field(default="*")

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.dashscope_api_key)

    @property
    def workspace_path(self) -> Path:
        p = Path(self.agent_workspace_dir)
        if not p.is_absolute():
            p = self.base_dir / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def rag_index_full_path(self) -> Path:
        p = Path(self.rag_index_path)
        if not p.is_absolute():
            p = self.base_dir / p
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def rag_docs_full_path(self) -> Path:
        p = Path(self.rag_docs_path)
        if not p.is_absolute():
            p = self.base_dir / p
        p.mkdir(parents=True, exist_ok=True)
        return p


@lru_cache
def get_settings() -> Settings:
    return Settings()
