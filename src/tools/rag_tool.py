from typing import Optional

from src.tools.base import ToolDefinition


def create_rag_tool(retriever) -> ToolDefinition:
    async def rag_search(question: str, top_k: int = 3):
        if not retriever or not retriever.is_ready:
            return {"status": "empty", "message": "Knowledge base is empty. Add documents to data/documents/."}
        context, sources = retriever.query(question, top_k)
        return {"context": context, "sources": sources}

    return ToolDefinition(
        name="rag_search",
        description="Search local knowledge base for documents relevant to the query.",
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question or query to search for"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 3)"},
            },
            "required": ["question"],
        },
        execute=rag_search,
        category="rag",
    )
