"""Tests for search engine."""

import pytest
from unittest.mock import AsyncMock, patch
from bereans.search.engine import SearchEngine
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
def engine(stores):
    sqlite, vector = stores
    return SearchEngine(sqlite_store=sqlite, vector_store=vector)


@pytest.fixture
def seeded_engine(engine, stores):
    sqlite, vector = stores
    doc_id = sqlite.insert_document(
        title="Genesis",
        source_type="file",
        source_path="/genesis.txt",
        file_format="txt",
        content_full="In the beginning God created the heavens and the earth. And the earth was without form.",
        content_hash="gen_hash",
        metadata={"tags": ["bible"]},
    )
    chunk_ids = sqlite.insert_chunks(doc_id, [
        {"content": "In the beginning God created the heavens and the earth.", "chunk_index": 0, "start_char": 0, "end_char": 55, "metadata": {}},
        {"content": "And the earth was without form.", "chunk_index": 1, "start_char": 56, "end_char": 86, "metadata": {}},
    ])
    vector.add_chunks(
        ids=chunk_ids,
        documents=["In the beginning God created the heavens and the earth.", "And the earth was without form."],
        embeddings=[[0.9, 0.1, 0.1], [0.1, 0.9, 0.1]],
        metadatas=[
            {"document_id": doc_id, "source_type": "file", "file_format": "txt", "chunk_index": 0},
            {"document_id": doc_id, "source_type": "file", "file_format": "txt", "chunk_index": 1},
        ],
    )
    return engine


@pytest.mark.asyncio
async def test_search_returns_results(seeded_engine):
    mock_embedder = AsyncMock()
    mock_embedder.embed_single = AsyncMock(return_value=[0.9, 0.1, 0.1])

    with patch.object(seeded_engine, "_get_embedder", return_value=mock_embedder):
        results = await seeded_engine.search("creation of the world")

    assert len(results["results"]) > 0
    assert results["results"][0]["document_title"] == "Genesis"


@pytest.mark.asyncio
async def test_search_empty_query(seeded_engine):
    results = await seeded_engine.search("")
    assert results["results"] == []


@pytest.mark.asyncio
async def test_search_with_filter(seeded_engine):
    mock_embedder = AsyncMock()
    mock_embedder.embed_single = AsyncMock(return_value=[0.5, 0.5, 0.5])

    with patch.object(seeded_engine, "_get_embedder", return_value=mock_embedder):
        results = await seeded_engine.search("anything", source_type="file")

    assert len(results["results"]) > 0
