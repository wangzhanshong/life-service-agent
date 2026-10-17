from fastapi import APIRouter, Depends

from src.server.dependencies import get_tool_registry

router = APIRouter()


@router.get("/tools")
async def list_tools(registry=Depends(get_tool_registry)):
    """List all available tools with their descriptions."""
    tools = []
    for tool in registry.list_all():
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "category": tool.category,
            "parameters": tool.parameters,
        })
    return {"count": len(tools), "categories": registry.get_categories(), "tools": tools}
