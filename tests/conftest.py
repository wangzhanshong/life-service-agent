import os
import sys
from pathlib import Path
from typing import AsyncGenerator

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base import Base
from src.tools.base import ToolRegistry
from src.tools.compute_tools import create_compute_tools
from src.tools.file_tools import create_file_tools
from src.tools.mcp_tools import create_mcp_tools


@pytest.fixture
def tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for tool in create_compute_tools():
        registry.register(tool)
    for tool in create_mcp_tools():
        registry.register(tool)
    workshop_dir = os.path.join(os.path.dirname(__file__), "test_workspace")
    os.makedirs(workshop_dir, exist_ok=True)
    return registry


@pytest.fixture
def test_workspace(tmp_path: Path) -> Path:
    ws = tmp_path / "workspace"
    ws.mkdir(exist_ok=True)
    return ws


@pytest.fixture
async def test_db(tmp_path: Path) -> AsyncGenerator[AsyncSession, None]:
    db_path = tmp_path / "test.db"
    db_url = f"sqlite+aiosqlite:///{db_path}"

    import src.core.config as config_mod
    import src.models.base as base_mod

    old_url = config_mod.get_settings().database_url
    config_mod.get_settings().database_url = db_url

    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    base_mod.engine = engine
    base_mod.async_session_factory = session_factory

    async with session_factory() as session:
        yield session

    await engine.dispose()
    config_mod.get_settings().database_url = old_url


@pytest.fixture
async def test_client(test_db: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    from src.server.app import create_app

    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
