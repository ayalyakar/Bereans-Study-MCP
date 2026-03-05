"""Tests for Ollama embedding client."""

import pytest
from unittest.mock import AsyncMock, patch
from bereans.embeddings.ollama import OllamaEmbedder


@pytest.fixture
def embedder():
    return OllamaEmbedder(host="http://localhost:11434", model="nomic-embed-text")


@pytest.mark.asyncio
async def test_embed_single_text(embedder):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"embeddings": [[0.1, 0.2, 0.3]]}
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        result = await embedder.embed(["hello world"])
        assert len(result) == 1
        assert len(result[0]) == 3


@pytest.mark.asyncio
async def test_embed_multiple_texts(embedder):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json = lambda: {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        result = await embedder.embed(["text one", "text two"])
        assert len(result) == 2


@pytest.mark.asyncio
async def test_embed_empty_list(embedder):
    result = await embedder.embed([])
    assert result == []


@pytest.mark.asyncio
async def test_health_check_success(embedder):
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = lambda: None

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        assert await embedder.health_check() is True


@pytest.mark.asyncio
async def test_health_check_failure(embedder):
    with patch("httpx.AsyncClient.get", side_effect=Exception("connection refused")):
        assert await embedder.health_check() is False
