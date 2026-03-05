"""Tests for SQLite storage layer."""

import pytest
from pathlib import Path
from bereans.storage.sqlite_store import SQLiteStore


@pytest.fixture
def store(tmp_path):
    db_path = tmp_path / "test.db"
    s = SQLiteStore(db_path)
    s.initialize()
    return s


def test_initialize_creates_tables(store):
    tables = store.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    table_names = {row[0] for row in tables}
    assert "documents" in table_names
    assert "chunks" in table_names


def test_insert_and_get_document(store):
    doc_id = store.insert_document(
        title="Genesis",
        source_type="file",
        source_path="/books/genesis.pdf",
        file_format="pdf",
        content_full="In the beginning God created the heavens and the earth.",
        content_hash="abc123",
        metadata={"author": "Moses", "tags": ["bible", "ot"]},
    )
    doc = store.get_document(doc_id)
    assert doc is not None
    assert doc["title"] == "Genesis"
    assert doc["source_type"] == "file"
    assert doc["content_full"].startswith("In the beginning")


def test_duplicate_hash_rejected(store):
    store.insert_document(
        title="Doc A", source_type="file", source_path="/a.txt",
        file_format="txt", content_full="Hello", content_hash="same_hash", metadata={},
    )
    result = store.insert_document(
        title="Doc B", source_type="file", source_path="/b.txt",
        file_format="txt", content_full="Hello", content_hash="same_hash", metadata={},
    )
    assert result is None


def test_insert_and_get_chunks(store):
    doc_id = store.insert_document(
        title="Test", source_type="file", source_path="/test.txt",
        file_format="txt", content_full="Full text here", content_hash="hash1", metadata={},
    )
    chunk_ids = store.insert_chunks(doc_id, [
        {"content": "chunk one", "chunk_index": 0, "start_char": 0, "end_char": 9, "metadata": {}},
        {"content": "chunk two", "chunk_index": 1, "start_char": 10, "end_char": 19, "metadata": {}},
    ])
    assert len(chunk_ids) == 2
    chunks = store.get_chunks_for_document(doc_id)
    assert len(chunks) == 2
    doc = store.get_document(doc_id)
    assert doc["chunk_count"] == 2


def test_delete_document_cascades(store):
    doc_id = store.insert_document(
        title="Delete Me", source_type="file", source_path="/del.txt",
        file_format="txt", content_full="gone", content_hash="del_hash", metadata={},
    )
    store.insert_chunks(doc_id, [
        {"content": "chunk", "chunk_index": 0, "start_char": 0, "end_char": 5, "metadata": {}},
    ])
    deleted_chunks = store.delete_document(doc_id)
    assert deleted_chunks == 1
    assert store.get_document(doc_id) is None


def test_list_documents(store):
    for i in range(5):
        store.insert_document(
            title=f"Doc {i}", source_type="file" if i % 2 == 0 else "url",
            source_path=f"/doc{i}", file_format="pdf",
            content_full=f"content {i}", content_hash=f"hash_{i}", metadata={},
        )
    all_docs = store.list_documents(limit=10)
    assert len(all_docs) == 5
    file_docs = store.list_documents(source_type="file", limit=10)
    assert len(file_docs) == 3


def test_get_stats(store):
    store.insert_document(
        title="Stats Doc", source_type="file", source_path="/stats.pdf",
        file_format="pdf", content_full="data", content_hash="stats_hash", metadata={},
    )
    stats = store.get_stats()
    assert stats["total_documents"] == 1
    assert stats["by_format"]["pdf"] == 1


def test_get_document_by_hash(store):
    store.insert_document(
        title="Hash Lookup", source_type="file", source_path="/hash.txt",
        file_format="txt", content_full="find me", content_hash="findable_hash", metadata={},
    )
    doc = store.get_document_by_hash("findable_hash")
    assert doc is not None
    assert doc["title"] == "Hash Lookup"
    missing = store.get_document_by_hash("nonexistent")
    assert missing is None
