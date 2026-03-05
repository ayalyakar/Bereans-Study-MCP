"""Tests for the ingestion pipeline."""

import pytest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from bereans.ingestion.pipeline import IngestionPipeline
from bereans.storage.sqlite_store import SQLiteStore
from bereans.storage.vector_store import VectorStore


@pytest.fixture
def stores(tmp_path):
    sqlite = SQLiteStore(tmp_path / "test.db")
    sqlite.initialize()
    vector = VectorStore(tmp_path / "chroma")
    vector.initialize()
    return sqlite, vector


@pytest.fixture
def pipeline(stores):
    sqlite, vector = stores
    return IngestionPipeline(sqlite_store=sqlite, vector_store=vector)


@pytest.mark.asyncio
async def test_ingest_text_file(pipeline, tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text("This is a sample document with enough text to be ingested properly.", encoding="utf-8")

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch.object(pipeline, "_get_embedder", return_value=mock_embedder):
        result = await pipeline.ingest_file(str(f))

    assert result["status"] == "success"
    assert result["chunks_created"] >= 1
    assert result["document_id"] is not None


@pytest.mark.asyncio
async def test_ingest_duplicate_rejected(pipeline, tmp_path):
    f = tmp_path / "dup.txt"
    f.write_text("Duplicate content here.", encoding="utf-8")

    mock_embedder = AsyncMock()
    mock_embedder.embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    with patch.object(pipeline, "_get_embedder", return_value=mock_embedder):
        result1 = await pipeline.ingest_file(str(f))
        result2 = await pipeline.ingest_file(str(f))

    assert result1["status"] == "success"
    assert result2["status"] == "duplicate"


@pytest.mark.asyncio
async def test_ingest_unsupported_format(pipeline, tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("whatever", encoding="utf-8")

    result = await pipeline.ingest_file(str(f))
    assert result["status"] == "error"
    assert "Unsupported" in result["message"]


@pytest.mark.asyncio
async def test_ingest_nonexistent_file(pipeline):
    result = await pipeline.ingest_file("/does/not/exist.txt")
    assert result["status"] == "error"
