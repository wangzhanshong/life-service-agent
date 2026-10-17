import json

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agent.types import EventType
from src.core.security import sanitize_input
from src.repository.conversation_repo import ConversationRepository
from src.server.dependencies import get_engine, get_repo

router = APIRouter()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000, description="User message")
    conversation_id: str | None = Field(default=None, description="Existing conversation ID to continue")


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    iterations: int
    success: bool


@router.post("/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    engine=Depends(get_engine),
    repo: ConversationRepository = Depends(get_repo),
):
    """Send a message and get the agent's response."""
    user_input = sanitize_input(req.message)

    # Create or continue conversation
    if req.conversation_id:
        conv = await repo.get_by_id(req.conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = await repo.create(title=user_input[:50])

    # Save user message
    await repo.add_message(conv.id, "user", user_input)

    # Execute agent
    result = await engine.execute(user_input)

    # Save assistant response
    if result.answer:
        await repo.add_message(conv.id, "assistant", result.answer)
    for tc in result.tool_calls:
        await repo.add_message(
            conv.id, "tool",
            content=tc.result,
            tool_name=tc.name,
            tool_calls={"name": tc.name, "arguments": tc.arguments},
        )

    return ChatResponse(
        conversation_id=conv.id,
        answer=result.answer,
        iterations=result.iterations,
        success=result.success,
    )


@router.get("/chat/stream")
async def chat_stream(
    message: str,
    conversation_id: str | None = None,
    engine=Depends(get_engine),
    repo: ConversationRepository = Depends(get_repo),
):
    """Stream the agent's Re-Act reasoning process via SSE."""
    user_input = sanitize_input(message)

    if conversation_id:
        conv = await repo.get_by_id(conversation_id)
        if not conv:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        conv = await repo.create(title=user_input[:50])

    await repo.add_message(conv.id, "user", user_input)

    async def event_generator():
        full_answer = ""
        tool_results = []

        async for event in engine.stream(user_input):
            yield {"data": event.model_dump_json()}

            if event.event_type == EventType.FINAL_ANSWER:
                full_answer = event.content
            elif event.event_type == EventType.OBSERVATION:
                tool_results.append({"name": event.tool_name, "result": event.tool_result})

        # Save assistant response
        if full_answer:
            await repo.add_message(conv.id, "assistant", full_answer)
        for tr in tool_results:
            await repo.add_message(conv.id, "tool", content=tr["result"], tool_name=tr["name"])

        yield {"data": json.dumps({"done": True, "conversation_id": conv.id})}

    return EventSourceResponse(event_generator())


@router.get("/conversations")
async def list_conversations(repo: ConversationRepository = Depends(get_repo)):
    """List all conversation history."""
    convs = await repo.list_all()
    return [
        {
            "id": c.id,
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat(),
        }
        for c in convs
    ]


@router.get("/conversations/{conv_id}")
async def get_conversation(conv_id: str, repo: ConversationRepository = Depends(get_repo)):
    """Get conversation details with messages."""
    conv = await repo.get_by_id(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await repo.get_messages(conv_id)
    return {
        "id": conv.id,
        "title": conv.title,
        "messages": [
            {
                "role": m.role,
                "content": m.content,
                "tool_calls": m.tool_calls,
                "tool_name": m.tool_name,
                "created_at": m.created_at.isoformat(),
            }
            for m in messages
        ],
    }


@router.delete("/conversations/{conv_id}")
async def delete_conversation(conv_id: str, repo: ConversationRepository = Depends(get_repo)):
    """Delete a conversation."""
    deleted = await repo.delete(conv_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conv_id}
