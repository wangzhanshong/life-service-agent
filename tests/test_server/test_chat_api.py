import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_endpoint(test_client: AsyncClient):
    response = await test_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "tools_registered" in data


@pytest.mark.asyncio
async def test_tools_endpoint(test_client: AsyncClient):
    response = await test_client.get("/api/v1/tools")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    assert "categories" in data


@pytest.mark.asyncio
async def test_conversations_empty(test_client: AsyncClient):
    response = await test_client.get("/api/v1/conversations")
    assert response.status_code == 200
    assert isinstance(response.json(), list)


@pytest.mark.asyncio
async def test_ready_endpoint(test_client: AsyncClient):
    response = await test_client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}
