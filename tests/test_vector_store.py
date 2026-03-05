"""Tests for ChromaDB vector store."""

import pytest
from bereans.storage.vector_store import VectorStore


@pytest.fixture
def vstore(tmp_path):
    store = VectorStore(persist_dir=tmp_path / "chroma")
    store.initialize()
    return store


def test_initialize_creates_collection(vstore):
    assert vstore.collection is not None
    assert vstore.collection.name == "knowledge_base"


def test_add_and_query(vstore):
    vstore.add_chunks(
        ids=["c1", "c2"],
        documents=["The earth revolves around the sun", "Water boils at 100 degrees"],
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        metadatas=[
            {"document_id": "doc1", "source_type": "file", "file_format": "txt", "chunk_index": 0},
            {"document_id": "doc1", "source_type": "file", "file_format": "txt", "chunk_index": 1},
        ],
    )
    results = vstore.query(
        query_embedding=[0.1, 0.2, 0.3],
        top_k=2,
    )
    assert len(results["ids"][0]) == 2
    assert results["ids"][0][0] == "c1"


def test_query_with_filter(vstore):
    vstore.add_chunks(
        ids=["c1", "c2"],
        documents=["Bible text", "Science text"],
        embeddings=[[0.1, 0.2], [0.9, 0.8]],
        metadatas=[
            {"document_id": "d1", "source_type": "file", "file_format": "pdf", "chunk_index": 0},
            {"document_id": "d2", "source_type": "url", "file_format": "html", "chunk_index": 0},
        ],
    )
    results = vstore.query(
        query_embedding=[0.1, 0.2],
        top_k=10,
        where={"source_type": "file"},
    )
    assert len(results["ids"][0]) == 1
    assert results["ids"][0][0] == "c1"


def test_delete_by_document_id(vstore):
    vstore.add_chunks(
        ids=["c1", "c2", "c3"],
        documents=["a", "b", "c"],
        embeddings=[[0.1], [0.2], [0.3]],
        metadatas=[
            {"document_id": "d1", "source_type": "file", "file_format": "txt", "chunk_index": 0},
            {"document_id": "d1", "source_type": "file", "file_format": "txt", "chunk_index": 1},
            {"document_id": "d2", "source_type": "file", "file_format": "txt", "chunk_index": 0},
        ],
    )
    deleted = vstore.delete_by_document_id("d1")
    assert deleted == 2
    assert vstore.collection.count() == 1


def test_count(vstore):
    assert vstore.count() == 0
    vstore.add_chunks(
        ids=["c1"],
        documents=["text"],
        embeddings=[[0.1]],
        metadatas=[{"document_id": "d1", "source_type": "file", "file_format": "txt", "chunk_index": 0}],
    )
    assert vstore.count() == 1
