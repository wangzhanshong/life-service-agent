from datetime import datetime
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.models.conversation import Conversation
from src.models.message import Message


class ConversationRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, title: str = "") -> Conversation:
        conv = Conversation(title=title[:200] if title else "New Conversation")
        self.session.add(conv)
        await self.session.flush()
        return conv

    async def get_by_id(self, conv_id: str) -> Optional[Conversation]:
        result = await self.session.execute(
            select(Conversation).where(Conversation.id == conv_id)
        )
        return result.scalar_one_or_none()

    async def list_all(self, limit: int = 50, offset: int = 0) -> List[Conversation]:
        result = await self.session.execute(
            select(Conversation)
            .order_by(Conversation.updated_at.desc())
            .offset(offset)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def update_title(self, conv_id: str, title: str) -> None:
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conv_id)
            .values(title=title[:200], updated_at=datetime.utcnow())
        )

    async def delete(self, conv_id: str) -> bool:
        conv = await self.get_by_id(conv_id)
        if conv is None:
            return False
        await self.session.delete(conv)
        return True

    async def add_message(
        self,
        conv_id: str,
        role: str,
        content: str = "",
        tool_calls: dict | None = None,
        tool_name: str | None = None,
    ) -> Message:
        msg = Message(
            conversation_id=conv_id,
            role=role,
            content=content,
            tool_calls=tool_calls,
            tool_name=tool_name,
        )
        self.session.add(msg)

        # Update conversation timestamp
        await self.session.execute(
            update(Conversation)
            .where(Conversation.id == conv_id)
            .values(updated_at=datetime.utcnow())
        )

        await self.session.flush()
        return msg

    async def get_messages(self, conv_id: str) -> List[Message]:
        result = await self.session.execute(
            select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at.asc())
        )
        return list(result.scalars().all())

    async def count(self) -> int:
        from sqlalchemy import func as sqlfunc
        result = await self.session.execute(select(sqlfunc.count()).select_from(Conversation))
        return result.scalar() or 0
