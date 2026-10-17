from fastapi import APIRouter, Depends

from src.llm.client import get_llm_client
from src.server.dependencies import get_rag, get_tool_registry

router = APIRouter()


@router.get("/health")
async def health_check(registry=Depends(get_tool_registry), rag=Depends(get_rag)):
    """Health check endpoint."""
    llm_available = False
    try:
        client = get_llm_client()
        llm_available = client.is_available
    except Exception:
        pass

    return {
        "status": "healthy",
        "version": "2.0.0",
        "llm_available": llm_available,
        "rag_documents": rag.vector_store.size if rag else 0,
        "tools_registered": registry.tool_count,
        "tool_categories": registry.get_categories(),
    }


@router.get("/ready")
async def readiness():
    """Readiness probe for Kubernetes."""
    return {"status": "ready"}
