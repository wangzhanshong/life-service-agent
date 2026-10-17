from typing import Optional

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.agent.react_engine import ReActEngine
from src.core.config import get_settings
from src.llm.client import get_llm_client
from src.models.base import get_async_session
from src.rag.document_processor import DocumentProcessor
from src.rag.embedding import EmbeddingService
from src.rag.retriever import RAGRetriever
from src.rag.vector_store import VectorStore
from src.repository.conversation_repo import ConversationRepository
from src.tools.base import ToolRegistry
from src.tools.compute_tools import create_compute_tools
from src.tools.file_tools import create_file_tools
from src.tools.mcp_tools import create_mcp_tools
from src.tools.rag_tool import create_rag_tool

# Singleton instances
_registry: Optional[ToolRegistry] = None
_rag_retriever: Optional[RAGRetriever] = None
_engine: Optional[ReActEngine] = None


def init_rag():
    """Initialize RAG system at startup."""
    global _rag_retriever
    settings = get_settings()
    embedding = EmbeddingService()
    vector_store = VectorStore(dimension=embedding.dimension, index_path=settings.rag_index_full_path)
    processor = DocumentProcessor(chunk_size=settings.rag_chunk_size, chunk_overlap=settings.rag_chunk_overlap)
    _rag_retriever = RAGRetriever(embedding, vector_store, processor, top_k=settings.rag_top_k)

    # Load existing index or build from documents
    if vector_store.load(settings.rag_index_full_path):
        pass
    elif settings.rag_docs_full_path.is_dir():
        count = _rag_retriever.build_knowledge_base(str(settings.rag_docs_full_path))
        if count > 0:
            _rag_retriever.save(settings.rag_index_full_path)


def get_tool_registry() -> ToolRegistry:
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
        for tool in create_file_tools():
            _registry.register(tool)
        for tool in create_compute_tools():
            _registry.register(tool)
        for tool in create_mcp_tools():
            _registry.register(tool)
        if _rag_retriever:
            _registry.register(create_rag_tool(_rag_retriever))
    return _registry


def get_rag() -> Optional[RAGRetriever]:
    return _rag_retriever


def get_engine() -> ReActEngine:
    global _engine
    if _engine is None:
        llm = get_llm_client()
        registry = get_tool_registry()
        _engine = ReActEngine(registry, llm, _rag_retriever)
    return _engine


async def get_repo(session: AsyncSession = Depends(get_async_session)) -> ConversationRepository:
    return ConversationRepository(session)
