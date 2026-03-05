# Bereans Study MCP Server — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Build a Python MCP server that ingests documents (PDF, EPUB, DOCX, HTML, Markdown, CSV, JSON, TXT), stores them in SQLite, embeds chunks via Ollama into ChromaDB, and exposes 6 tools to Claude Desktop for semantic search and knowledge management.

**Architecture:** SQLite stores full documents and metadata. ChromaDB stores chunked embeddings for semantic search. Ollama (nomic-embed-text) generates embeddings locally. Modular parsers handle each file format. MCP server communicates with Claude Desktop via stdio.

**Tech Stack:** Python 3.13, mcp[cli], chromadb, pymupdf, ebooklib, python-docx, beautifulsoup4, pandas, httpx, python-dotenv, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/bereans/__init__.py`
- Create: `src/bereans/config.py`
- Create: `tests/__init__.py`

**Step 1: Create pyproject.toml**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "bereans-study-mcp"
version = "0.1.0"
description = "MCP server — a living knowledge base with semantic search"
requires-python = ">=3.11"
dependencies = [
    "mcp[cli]>=1.0.0",
    "chromadb>=0.5.0",
    "httpx>=0.27.0",
    "python-dotenv>=1.0.0",
    "pymupdf>=1.24.0",
    "ebooklib>=0.18",
    "python-docx>=1.1.0",
    "beautifulsoup4>=4.12.0",
    "lxml>=5.0.0",
    "pandas>=2.2.0",
    "tiktoken>=0.7.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
]

[tool.hatch.build.targets.wheel]
packages = ["src/bereans"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
```

**Step 2: Create .env.example**

```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
SQLITE_PATH=./data/bereans.db
CHROMA_PATH=./data/chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

**Step 3: Create .gitignore**

```
data/
*.db
__pycache__/
*.pyc
.env
*.egg-info/
dist/
build/
.venv/
```

**Step 4: Create src/bereans/__init__.py**

```python
"""Bereans Study MCP Server — a living knowledge base."""
```

**Step 5: Create src/bereans/config.py**

```python
"""Configuration loaded from environment variables with sensible defaults."""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "nomic-embed-text")

SQLITE_PATH = Path(os.getenv("SQLITE_PATH", str(BASE_DIR / "data" / "bereans.db")))
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", str(BASE_DIR / "data" / "chroma")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

SUPPORTED_FORMATS = {
    "pdf", "epub", "mobi", "docx", "txt", "md",
    "html", "htm", "csv", "json",
}
```

**Step 6: Create tests/__init__.py**

```python
```

**Step 7: Install dependencies**

Run: `cd "D:/Bereans Study MCP" && pip install -e ".[dev]"`
Expected: All dependencies install successfully.

**Step 8: Verify config loads**

Run: `cd "D:/Bereans Study MCP" && python -c "from bereans.config import OLLAMA_HOST, SQLITE_PATH; print(OLLAMA_HOST, SQLITE_PATH)"`
Expected: `http://localhost:11434 <path>/data/bereans.db`

**Step 9: Commit**

```bash
git add pyproject.toml .env.example .gitignore src/ tests/
git commit -m "feat: project scaffolding with config and dependencies"
```

---

### Task 2: SQLite Storage Layer

**Files:**
- Create: `src/bereans/storage/__init__.py`
- Create: `src/bereans/storage/sqlite_store.py`
- Create: `tests/test_storage.py`

**Step 1: Create src/bereans/storage/__init__.py**

```python
```

**Step 2: Write the failing tests**

Create `tests/test_storage.py`:

```python
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
        title="Doc A",
        source_type="file",
        source_path="/a.txt",
        file_format="txt",
        content_full="Hello",
        content_hash="same_hash",
        metadata={},
    )
    result = store.insert_document(
        title="Doc B",
        source_type="file",
        source_path="/b.txt",
        file_format="txt",
        content_full="Hello",
        content_hash="same_hash",
        metadata={},
    )
    assert result is None  # duplicate rejected


def test_insert_and_get_chunks(store):
    doc_id = store.insert_document(
        title="Test",
        source_type="file",
        source_path="/test.txt",
        file_format="txt",
        content_full="Full text here",
        content_hash="hash1",
        metadata={},
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
        title="Delete Me",
        source_type="file",
        source_path="/del.txt",
        file_format="txt",
        content_full="gone",
        content_hash="del_hash",
        metadata={},
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
            title=f"Doc {i}",
            source_type="file" if i % 2 == 0 else "url",
            source_path=f"/doc{i}",
            file_format="pdf",
            content_full=f"content {i}",
            content_hash=f"hash_{i}",
            metadata={},
        )
    all_docs = store.list_documents(limit=10)
    assert len(all_docs) == 5

    file_docs = store.list_documents(source_type="file", limit=10)
    assert len(file_docs) == 3


def test_get_stats(store):
    store.insert_document(
        title="Stats Doc",
        source_type="file",
        source_path="/stats.pdf",
        file_format="pdf",
        content_full="data",
        content_hash="stats_hash",
        metadata={},
    )
    stats = store.get_stats()
    assert stats["total_documents"] == 1
    assert stats["by_format"]["pdf"] == 1


def test_get_document_by_hash(store):
    store.insert_document(
        title="Hash Lookup",
        source_type="file",
        source_path="/hash.txt",
        file_format="txt",
        content_full="find me",
        content_hash="findable_hash",
        metadata={},
    )
    doc = store.get_document_by_hash("findable_hash")
    assert doc is not None
    assert doc["title"] == "Hash Lookup"

    missing = store.get_document_by_hash("nonexistent")
    assert missing is None
```

**Step 3: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_storage.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'bereans.storage.sqlite_store'`

**Step 4: Write the SQLite store implementation**

Create `src/bereans/storage/sqlite_store.py`:

```python
"""SQLite storage for documents and chunks."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


class SQLiteStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._conn: sqlite3.Connection | None = None

    def initialize(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS documents (
                id              TEXT PRIMARY KEY,
                title           TEXT,
                source_type     TEXT NOT NULL,
                source_path     TEXT,
                file_format     TEXT,
                content_full    TEXT,
                content_hash    TEXT UNIQUE,
                metadata        TEXT DEFAULT '{}',
                chunk_count     INTEGER DEFAULT 0,
                created_at      TEXT,
                updated_at      TEXT
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id              TEXT PRIMARY KEY,
                document_id     TEXT NOT NULL,
                chunk_index     INTEGER,
                content         TEXT,
                start_char      INTEGER,
                end_char        INTEGER,
                metadata        TEXT DEFAULT '{}',
                FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);
            CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
            CREATE INDEX IF NOT EXISTS idx_documents_source_type ON documents(source_type);
            CREATE INDEX IF NOT EXISTS idx_documents_format ON documents(file_format);
        """)
        self._conn.commit()

    def execute(self, sql: str, params=()):
        return self._conn.execute(sql, params)

    def insert_document(
        self,
        title: str,
        source_type: str,
        source_path: str,
        file_format: str,
        content_full: str,
        content_hash: str,
        metadata: dict,
    ) -> str | None:
        doc_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        try:
            self._conn.execute(
                """INSERT INTO documents
                   (id, title, source_type, source_path, file_format,
                    content_full, content_hash, metadata, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (doc_id, title, source_type, source_path, file_format,
                 content_full, content_hash, json.dumps(metadata), now, now),
            )
            self._conn.commit()
            return doc_id
        except sqlite3.IntegrityError:
            return None

    def get_document(self, doc_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_document_by_hash(self, content_hash: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def insert_chunks(self, document_id: str, chunks: list[dict]) -> list[str]:
        chunk_ids = []
        for chunk in chunks:
            chunk_id = uuid.uuid4().hex
            self._conn.execute(
                """INSERT INTO chunks
                   (id, document_id, chunk_index, content, start_char, end_char, metadata)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (chunk_id, document_id, chunk["chunk_index"], chunk["content"],
                 chunk["start_char"], chunk["end_char"], json.dumps(chunk.get("metadata", {}))),
            )
            chunk_ids.append(chunk_id)
        self._conn.execute(
            "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE id = ?",
            (len(chunks), datetime.now(timezone.utc).isoformat(), document_id),
        )
        self._conn.commit()
        return chunk_ids

    def get_chunks_for_document(self, document_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> int:
        chunk_count = self._conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE document_id = ?", (doc_id,)
        ).fetchone()[0]
        self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (doc_id,))
        self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        self._conn.commit()
        return chunk_count

    def list_documents(
        self,
        source_type: str | None = None,
        file_format: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        query = "SELECT id, title, source_type, source_path, file_format, chunk_count, created_at FROM documents"
        conditions = []
        params = []
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)
        if file_format:
            conditions.append("file_format = ?")
            params.append(file_format)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def count_documents(
        self,
        source_type: str | None = None,
        file_format: str | None = None,
    ) -> int:
        query = "SELECT COUNT(*) FROM documents"
        conditions = []
        params = []
        if source_type:
            conditions.append("source_type = ?")
            params.append(source_type)
        if file_format:
            conditions.append("file_format = ?")
            params.append(file_format)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        return self._conn.execute(query, params).fetchone()[0]

    def get_stats(self) -> dict:
        total_docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        total_chunks = self._conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

        by_format = {}
        for row in self._conn.execute(
            "SELECT file_format, COUNT(*) FROM documents GROUP BY file_format"
        ).fetchall():
            by_format[row[0]] = row[1]

        by_source = {}
        for row in self._conn.execute(
            "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"
        ).fetchall():
            by_source[row[0]] = row[1]

        last = self._conn.execute(
            "SELECT created_at FROM documents ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

        db_size = self.db_path.stat().st_size / (1024 * 1024) if self.db_path.exists() else 0

        return {
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "by_format": by_format,
            "by_source_type": by_source,
            "storage_size_mb": round(db_size, 2),
            "last_ingested": last[0] if last else None,
        }

    def close(self):
        if self._conn:
            self._conn.close()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        if "metadata" in d and isinstance(d["metadata"], str):
            d["metadata"] = json.loads(d["metadata"])
        return d
```

**Step 5: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_storage.py -v`
Expected: All 8 tests PASS.

**Step 6: Commit**

```bash
git add src/bereans/storage/ tests/test_storage.py
git commit -m "feat: SQLite storage layer with full CRUD and stats"
```

---

### Task 3: Ollama Embedding Client

**Files:**
- Create: `src/bereans/embeddings/__init__.py`
- Create: `src/bereans/embeddings/ollama.py`
- Create: `tests/test_embeddings.py`

**Step 1: Create src/bereans/embeddings/__init__.py**

```python
```

**Step 2: Write the failing tests**

Create `tests/test_embeddings.py`:

```python
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
```

**Step 3: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_embeddings.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write the Ollama embedder**

Create `src/bereans/embeddings/ollama.py`:

```python
"""Ollama API client for generating embeddings."""

import httpx


class OllamaEmbedder:
    def __init__(self, host: str, model: str):
        self.host = host.rstrip("/")
        self.model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/embed",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            data = response.json()
            return data["embeddings"]

    async def embed_single(self, text: str) -> list[float]:
        result = await self.embed([text])
        return result[0]

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(self.host)
                response.raise_for_status()
                return True
        except Exception:
            return False
```

**Step 5: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_embeddings.py -v`
Expected: All 5 tests PASS.

**Step 6: Commit**

```bash
git add src/bereans/embeddings/ tests/test_embeddings.py
git commit -m "feat: Ollama embedding client with health check"
```

---

### Task 4: ChromaDB Vector Store

**Files:**
- Create: `src/bereans/storage/vector_store.py`
- Create: `tests/test_vector_store.py`

**Step 1: Write the failing tests**

Create `tests/test_vector_store.py`:

```python
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
    assert results["ids"][0][0] == "c1"  # closest match


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
```

**Step 2: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_vector_store.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the ChromaDB vector store**

Create `src/bereans/storage/vector_store.py`:

```python
"""ChromaDB vector store for chunk embeddings and semantic search."""

from pathlib import Path
import chromadb


class VectorStore:
    def __init__(self, persist_dir: Path):
        self.persist_dir = Path(persist_dir)
        self.client: chromadb.ClientAPI | None = None
        self.collection: chromadb.Collection | None = None

    def initialize(self):
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        where: dict | None = None,
    ) -> dict:
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def delete_by_document_id(self, document_id: str) -> int:
        existing = self.collection.get(
            where={"document_id": document_id},
            include=[],
        )
        count = len(existing["ids"])
        if count > 0:
            self.collection.delete(ids=existing["ids"])
        return count

    def count(self) -> int:
        return self.collection.count()
```

**Step 4: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_vector_store.py -v`
Expected: All 5 tests PASS.

**Step 5: Commit**

```bash
git add src/bereans/storage/vector_store.py tests/test_vector_store.py
git commit -m "feat: ChromaDB vector store with query, filter, and delete"
```

---

### Task 5: Text Chunker

**Files:**
- Create: `src/bereans/ingestion/__init__.py`
- Create: `src/bereans/ingestion/chunker.py`
- Create: `tests/test_chunker.py`

**Step 1: Create src/bereans/ingestion/__init__.py**

```python
```

**Step 2: Write the failing tests**

Create `tests/test_chunker.py`:

```python
"""Tests for smart text chunker."""

import pytest
from bereans.ingestion.chunker import TextChunker


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=50, chunk_overlap=10)


def test_short_text_single_chunk(chunker):
    text = "This is a short text."
    chunks = chunker.chunk(text)
    assert len(chunks) == 1
    assert chunks[0]["content"] == text
    assert chunks[0]["start_char"] == 0
    assert chunks[0]["end_char"] == len(text)


def test_long_text_multiple_chunks(chunker):
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    assert len(chunks) > 1
    # Every chunk should have content
    for chunk in chunks:
        assert len(chunk["content"]) > 0
        assert chunk["start_char"] < chunk["end_char"]


def test_chunks_have_overlap(chunker):
    sentences = [f"Sentence number {i} is here." for i in range(20)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    if len(chunks) >= 2:
        # Second chunk should start before first chunk ends (overlap)
        assert chunks[1]["start_char"] < chunks[0]["end_char"]


def test_chunk_indices_sequential(chunker):
    sentences = [f"Sentence {i}." for i in range(30)]
    text = " ".join(sentences)
    chunks = chunker.chunk(text)
    for i, chunk in enumerate(chunks):
        assert chunk["chunk_index"] == i


def test_empty_text(chunker):
    chunks = chunker.chunk("")
    assert chunks == []


def test_respects_paragraph_boundaries():
    chunker = TextChunker(chunk_size=30, chunk_overlap=5)
    text = "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here."
    chunks = chunker.chunk(text)
    # Should prefer splitting at paragraph boundaries
    assert len(chunks) >= 2
```

**Step 3: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_chunker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write the chunker**

Create `src/bereans/ingestion/chunker.py`:

```python
"""Smart text chunker that respects sentence and paragraph boundaries."""

import re
import tiktoken


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = tiktoken.get_encoding("cl100k_base")

    def chunk(self, text: str) -> list[dict]:
        if not text or not text.strip():
            return []

        token_count = len(self._encoder.encode(text))
        if token_count <= self.chunk_size:
            return [{
                "content": text,
                "chunk_index": 0,
                "start_char": 0,
                "end_char": len(text),
                "metadata": {},
            }]

        segments = self._split_into_segments(text)
        return self._merge_segments(segments, text)

    def _split_into_segments(self, text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", text)
        segments = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            para_tokens = len(self._encoder.encode(para))
            if para_tokens <= self.chunk_size:
                segments.append(para)
            else:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                segments.extend(s.strip() for s in sentences if s.strip())
        return segments

    def _merge_segments(self, segments: list[str], original_text: str) -> list[dict]:
        chunks = []
        current_segments = []
        current_tokens = 0

        for segment in segments:
            seg_tokens = len(self._encoder.encode(segment))

            if current_tokens + seg_tokens > self.chunk_size and current_segments:
                chunk_text = "\n\n".join(current_segments)
                start_char = original_text.find(current_segments[0])
                end_char = original_text.find(current_segments[-1]) + len(current_segments[-1])
                chunks.append({
                    "content": chunk_text,
                    "chunk_index": len(chunks),
                    "start_char": max(0, start_char),
                    "end_char": min(len(original_text), end_char),
                    "metadata": {},
                })

                # Keep overlap segments
                overlap_segments = []
                overlap_tokens = 0
                for s in reversed(current_segments):
                    s_tokens = len(self._encoder.encode(s))
                    if overlap_tokens + s_tokens > self.chunk_overlap:
                        break
                    overlap_segments.insert(0, s)
                    overlap_tokens += s_tokens
                current_segments = overlap_segments
                current_tokens = overlap_tokens

            current_segments.append(segment)
            current_tokens += seg_tokens

        if current_segments:
            chunk_text = "\n\n".join(current_segments)
            start_char = original_text.find(current_segments[0])
            end_char = original_text.find(current_segments[-1]) + len(current_segments[-1])
            chunks.append({
                "content": chunk_text,
                "chunk_index": len(chunks),
                "start_char": max(0, start_char),
                "end_char": min(len(original_text), end_char),
                "metadata": {},
            })

        return chunks

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))
```

**Step 5: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_chunker.py -v`
Expected: All 6 tests PASS.

**Step 6: Commit**

```bash
git add src/bereans/ingestion/ tests/test_chunker.py
git commit -m "feat: smart text chunker with paragraph and sentence awareness"
```

---

### Task 6: File Parsers

**Files:**
- Create: `src/bereans/ingestion/parsers/__init__.py`
- Create: `src/bereans/ingestion/parsers/plaintext.py`
- Create: `src/bereans/ingestion/parsers/markdown.py`
- Create: `src/bereans/ingestion/parsers/pdf.py`
- Create: `src/bereans/ingestion/parsers/epub.py`
- Create: `src/bereans/ingestion/parsers/docx.py`
- Create: `src/bereans/ingestion/parsers/html.py`
- Create: `src/bereans/ingestion/parsers/csv_json.py`
- Create: `tests/test_parsers.py`

**Step 1: Write the parser interface and registry**

Create `src/bereans/ingestion/parsers/__init__.py`:

```python
"""Parser registry — maps file formats to parser classes."""

from dataclasses import dataclass, field


@dataclass
class ParseResult:
    text: str
    metadata: dict = field(default_factory=dict)
    title: str | None = None


class BaseParser:
    """All parsers implement this interface."""

    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        raise NotImplementedError


_REGISTRY: dict[str, type[BaseParser]] = {}


def register(fmt: str):
    """Decorator to register a parser for a file format."""
    def decorator(cls):
        _REGISTRY[fmt] = cls
        return cls
    return decorator


def get_parser(fmt: str) -> BaseParser:
    fmt = fmt.lower().lstrip(".")
    if fmt not in _REGISTRY:
        raise ValueError(f"Unsupported format: {fmt}. Supported: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[fmt]()


def supported_formats() -> list[str]:
    return sorted(_REGISTRY.keys())
```

**Step 2: Create the plaintext parser**

Create `src/bereans/ingestion/parsers/plaintext.py`:

```python
"""Plain text parser."""

from pathlib import Path
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("txt")
class PlainTextParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        title = Path(source_path or source).stem if isinstance(source, str) else None
        return ParseResult(text=text, title=title)
```

**Step 3: Create the markdown parser**

Create `src/bereans/ingestion/parsers/markdown.py`:

```python
"""Markdown parser — preserves structure, extracts title from first heading."""

import re
from pathlib import Path
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("md")
class MarkdownParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            text = Path(source).read_text(encoding="utf-8", errors="replace")
        title = None
        first_heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
        if first_heading:
            title = first_heading.group(1).strip()
        return ParseResult(text=text, title=title or Path(source_path or source).stem)
```

**Step 4: Create the PDF parser**

Create `src/bereans/ingestion/parsers/pdf.py`:

```python
"""PDF parser using PyMuPDF."""

from pathlib import Path
import pymupdf
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("pdf")
class PDFParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            doc = pymupdf.open(stream=source, filetype="pdf")
        else:
            doc = pymupdf.open(source)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                pages.append(text)
        full_text = "\n\n".join(pages)
        metadata = {"page_count": len(doc)}
        title = doc.metadata.get("title") or Path(source_path or str(source)).stem
        doc.close()
        return ParseResult(text=full_text, metadata=metadata, title=title)
```

**Step 5: Create the EPUB parser**

Create `src/bereans/ingestion/parsers/epub.py`:

```python
"""EPUB parser using ebooklib + BeautifulSoup."""

from pathlib import Path
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("epub")
class EPUBParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        book = epub.read_epub(source, options={"ignore_ncx": True})
        chapters = []
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), "lxml")
            text = soup.get_text(separator="\n", strip=True)
            if text.strip():
                chapters.append(text)
        full_text = "\n\n".join(chapters)
        title = book.get_metadata("DC", "title")
        title_str = title[0][0] if title else Path(source_path or str(source)).stem
        metadata = {}
        authors = book.get_metadata("DC", "creator")
        if authors:
            metadata["author"] = authors[0][0]
        return ParseResult(text=full_text, metadata=metadata, title=title_str)
```

**Step 6: Create the DOCX parser**

Create `src/bereans/ingestion/parsers/docx.py`:

```python
"""DOCX parser using python-docx."""

from pathlib import Path
from docx import Document
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("docx")
class DOCXParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        doc = Document(source)
        paragraphs = []
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)
        full_text = "\n\n".join(paragraphs)
        title = doc.core_properties.title or Path(source_path or str(source)).stem
        metadata = {}
        if doc.core_properties.author:
            metadata["author"] = doc.core_properties.author
        return ParseResult(text=full_text, metadata=metadata, title=title)
```

**Step 7: Create the HTML parser**

Create `src/bereans/ingestion/parsers/html.py`:

```python
"""HTML parser using BeautifulSoup."""

from pathlib import Path
from bs4 import BeautifulSoup
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("html")
@register("htm")
class HTMLParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            html = source.decode("utf-8", errors="replace")
        elif Path(source).exists():
            html = Path(source).read_text(encoding="utf-8", errors="replace")
        else:
            html = source
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else None
        text = soup.get_text(separator="\n", strip=True)
        return ParseResult(
            text=text,
            title=title or Path(source_path).stem if source_path else title,
        )
```

**Step 8: Create the CSV/JSON parser**

Create `src/bereans/ingestion/parsers/csv_json.py`:

```python
"""CSV and JSON parsers."""

import json as json_lib
from pathlib import Path
import pandas as pd
from bereans.ingestion.parsers import BaseParser, ParseResult, register


@register("csv")
class CSVParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            from io import BytesIO
            df = pd.read_csv(BytesIO(source))
        else:
            df = pd.read_csv(source)
        rows = []
        for _, row in df.iterrows():
            row_text = " | ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            rows.append(row_text)
        full_text = "\n".join(rows)
        metadata = {"columns": list(df.columns), "row_count": len(df)}
        title = Path(source_path or str(source)).stem if isinstance(source, str) else "csv_data"
        return ParseResult(text=full_text, metadata=metadata, title=title)


@register("json")
class JSONParser(BaseParser):
    def parse(self, source: str | bytes, source_path: str = "") -> ParseResult:
        if isinstance(source, bytes):
            data = json_lib.loads(source.decode("utf-8"))
        elif Path(source).exists():
            data = json_lib.loads(Path(source).read_text(encoding="utf-8"))
        else:
            data = json_lib.loads(source)
        text = self._flatten(data)
        title = Path(source_path or str(source)).stem if isinstance(source, str) else "json_data"
        return ParseResult(text=text, title=title)

    def _flatten(self, data, prefix: str = "") -> str:
        lines = []
        if isinstance(data, dict):
            for key, value in data.items():
                path = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(self._flatten(value, path))
                else:
                    lines.append(f"{path}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(self._flatten(item, f"{prefix}[{i}]"))
        else:
            lines.append(f"{prefix}: {data}" if prefix else str(data))
        return "\n".join(lines)
```

**Step 9: Write parser tests**

Create `tests/test_parsers.py`:

```python
"""Tests for file parsers."""

import json
import pytest
from bereans.ingestion.parsers import get_parser, supported_formats, ParseResult


def test_supported_formats():
    fmts = supported_formats()
    assert "txt" in fmts
    assert "md" in fmts
    assert "pdf" in fmts
    assert "html" in fmts
    assert "csv" in fmts
    assert "json" in fmts


def test_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        get_parser("xyz")


def test_plaintext_parser(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("Hello world, this is a test.", encoding="utf-8")
    parser = get_parser("txt")
    result = parser.parse(str(f), source_path=str(f))
    assert "Hello world" in result.text
    assert result.title == "test"


def test_markdown_parser(tmp_path):
    f = tmp_path / "doc.md"
    f.write_text("# My Title\n\nSome paragraph text.", encoding="utf-8")
    parser = get_parser("md")
    result = parser.parse(str(f), source_path=str(f))
    assert result.title == "My Title"
    assert "Some paragraph text" in result.text


def test_html_parser(tmp_path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><head><title>Page Title</title></head>"
        "<body><p>Content here</p><script>bad()</script></body></html>",
        encoding="utf-8",
    )
    parser = get_parser("html")
    result = parser.parse(str(f), source_path=str(f))
    assert result.title == "Page Title"
    assert "Content here" in result.text
    assert "bad()" not in result.text


def test_csv_parser(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age\nAlice,30\nBob,25", encoding="utf-8")
    parser = get_parser("csv")
    result = parser.parse(str(f), source_path=str(f))
    assert "Alice" in result.text
    assert "Bob" in result.text
    assert result.metadata["row_count"] == 2


def test_json_parser(tmp_path):
    f = tmp_path / "data.json"
    data = {"name": "Genesis", "chapters": 50, "author": "Moses"}
    f.write_text(json.dumps(data), encoding="utf-8")
    parser = get_parser("json")
    result = parser.parse(str(f), source_path=str(f))
    assert "Genesis" in result.text
    assert "Moses" in result.text
```

**Step 10: Run tests**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_parsers.py -v`
Expected: All 7 tests PASS.

**Step 11: Commit**

```bash
git add src/bereans/ingestion/parsers/ tests/test_parsers.py
git commit -m "feat: modular file parsers for txt, md, pdf, epub, docx, html, csv, json"
```

---

### Task 7: Ingestion Pipeline

**Files:**
- Create: `src/bereans/ingestion/pipeline.py`
- Create: `tests/test_ingestion.py`

**Step 1: Write the failing tests**

Create `tests/test_ingestion.py`:

```python
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
```

**Step 2: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_ingestion.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write the ingestion pipeline**

Create `src/bereans/ingestion/pipeline.py`:

```python
"""Ingestion pipeline — orchestrates parsing, chunking, embedding, and storage."""

import hashlib
from pathlib import Path

import httpx

from bereans.config import OLLAMA_HOST, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from bereans.embeddings.ollama import OllamaEmbedder
from bereans.ingestion.chunker import TextChunker
from bereans.ingestion.parsers import get_parser, supported_formats
from bereans.storage.sqlite_store import SQLiteStore
from bereans.storage.vector_store import VectorStore


class IngestionPipeline:
    def __init__(self, sqlite_store: SQLiteStore, vector_store: VectorStore):
        self.sqlite = sqlite_store
        self.vector = vector_store
        self.chunker = TextChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    def _get_embedder(self) -> OllamaEmbedder:
        return OllamaEmbedder(host=OLLAMA_HOST, model=OLLAMA_MODEL)

    async def ingest_file(
        self,
        file_path: str,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        path = Path(file_path)

        if not path.exists():
            return {"status": "error", "message": f"File not found: {file_path}"}

        fmt = path.suffix.lstrip(".").lower()
        try:
            parser = get_parser(fmt)
        except ValueError as e:
            return {"status": "error", "message": str(e)}

        try:
            result = parser.parse(str(path), source_path=str(path))
        except Exception as e:
            return {"status": "error", "message": f"Parse error: {e}"}

        if not result.text.strip():
            return {"status": "error", "message": "No extractable text found"}

        return await self._store(
            text=result.text,
            title=title or result.title or path.stem,
            source_type="file",
            source_path=str(path),
            file_format=fmt,
            metadata={**result.metadata, **({"tags": tags} if tags else {})},
        )

    async def ingest_url(
        self,
        url: str,
        title: str | None = None,
        tags: list[str] | None = None,
    ) -> dict:
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                response = await client.get(url)
                response.raise_for_status()
        except Exception as e:
            # Retry once
            try:
                async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                    response = await client.get(url)
                    response.raise_for_status()
            except Exception as e2:
                return {"status": "error", "message": f"URL unreachable: {e2}"}

        content_type = response.headers.get("content-type", "")
        content = response.content

        if "application/pdf" in content_type or url.lower().endswith(".pdf"):
            fmt = "pdf"
        elif url.lower().endswith(".epub"):
            fmt = "epub"
        elif url.lower().endswith(".docx"):
            fmt = "docx"
        elif url.lower().endswith((".csv",)):
            fmt = "csv"
        elif url.lower().endswith(".json") or "application/json" in content_type:
            fmt = "json"
        else:
            fmt = "html"

        try:
            parser = get_parser(fmt)
            result = parser.parse(content, source_path=url)
        except Exception as e:
            return {"status": "error", "message": f"Parse error: {e}"}

        if not result.text.strip():
            return {"status": "error", "message": "No extractable text found at URL"}

        return await self._store(
            text=result.text,
            title=title or result.title or url,
            source_type="url",
            source_path=url,
            file_format=fmt,
            metadata={**result.metadata, **({"tags": tags} if tags else {})},
        )

    async def _store(
        self,
        text: str,
        title: str,
        source_type: str,
        source_path: str,
        file_format: str,
        metadata: dict,
    ) -> dict:
        content_hash = hashlib.sha256(text.encode()).hexdigest()

        existing = self.sqlite.get_document_by_hash(content_hash)
        if existing:
            return {
                "status": "duplicate",
                "message": "Document already exists",
                "document_id": existing["id"],
            }

        doc_id = self.sqlite.insert_document(
            title=title,
            source_type=source_type,
            source_path=source_path,
            file_format=file_format,
            content_full=text,
            content_hash=content_hash,
            metadata=metadata,
        )

        if doc_id is None:
            return {"status": "error", "message": "Failed to store document (possible duplicate)"}

        try:
            chunks = self.chunker.chunk(text)
            chunk_ids = self.sqlite.insert_chunks(doc_id, chunks)

            embedder = self._get_embedder()
            chunk_texts = [c["content"] for c in chunks]
            embeddings = await embedder.embed(chunk_texts)

            self.vector.add_chunks(
                ids=chunk_ids,
                documents=chunk_texts,
                embeddings=embeddings,
                metadatas=[
                    {
                        "document_id": doc_id,
                        "source_type": source_type,
                        "file_format": file_format,
                        "chunk_index": c["chunk_index"],
                    }
                    for c in chunks
                ],
            )

            return {
                "status": "success",
                "document_id": doc_id,
                "chunks_created": len(chunks),
                "format_detected": file_format,
                "title": title,
            }
        except Exception as e:
            # Rollback on failure
            self.sqlite.delete_document(doc_id)
            return {"status": "error", "message": f"Ingestion failed (rolled back): {e}"}
```

**Step 4: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_ingestion.py -v`
Expected: All 4 tests PASS.

**Step 5: Commit**

```bash
git add src/bereans/ingestion/pipeline.py tests/test_ingestion.py
git commit -m "feat: ingestion pipeline with atomic rollback and deduplication"
```

---

### Task 8: Search Engine

**Files:**
- Create: `src/bereans/search/__init__.py`
- Create: `src/bereans/search/engine.py`
- Create: `tests/test_search.py`

**Step 1: Create src/bereans/search/__init__.py**

```python
```

**Step 2: Write the failing tests**

Create `tests/test_search.py`:

```python
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
```

**Step 3: Run tests to verify they fail**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_search.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write the search engine**

Create `src/bereans/search/engine.py`:

```python
"""Search engine — embeds queries, searches ChromaDB, enriches from SQLite."""

from bereans.config import OLLAMA_HOST, OLLAMA_MODEL
from bereans.embeddings.ollama import OllamaEmbedder
from bereans.storage.sqlite_store import SQLiteStore
from bereans.storage.vector_store import VectorStore


class SearchEngine:
    def __init__(self, sqlite_store: SQLiteStore, vector_store: VectorStore):
        self.sqlite = sqlite_store
        self.vector = vector_store

    def _get_embedder(self) -> OllamaEmbedder:
        return OllamaEmbedder(host=OLLAMA_HOST, model=OLLAMA_MODEL)

    async def search(
        self,
        query: str,
        top_k: int = 10,
        source_type: str | None = None,
        file_format: str | None = None,
    ) -> dict:
        if not query or not query.strip():
            return {"results": [], "total_sources_searched": 0}

        embedder = self._get_embedder()
        query_embedding = await embedder.embed_single(query)

        where = {}
        if source_type:
            where["source_type"] = source_type
        if file_format:
            where["file_format"] = file_format

        chroma_results = self.vector.query(
            query_embedding=query_embedding,
            top_k=top_k,
            where=where if where else None,
        )

        results = []
        seen_docs = set()
        for i, chunk_id in enumerate(chroma_results["ids"][0]):
            metadata = chroma_results["metadatas"][0][i]
            document_id = metadata.get("document_id")
            chunk_text = chroma_results["documents"][0][i]
            distance = chroma_results["distances"][0][i]

            doc = self.sqlite.get_document(document_id)
            if doc is None:
                continue

            seen_docs.add(document_id)

            # Get surrounding context from full text
            context = self._get_context(doc["content_full"], chunk_text)

            results.append({
                "chunk_text": chunk_text,
                "document_title": doc["title"],
                "source_path": doc["source_path"],
                "relevance_score": round(1 - distance, 4),
                "context": context,
                "document_id": document_id,
                "chunk_index": metadata.get("chunk_index", 0),
            })

        total_sources = self.sqlite.count_documents(
            source_type=source_type, file_format=file_format
        )

        return {
            "results": results,
            "total_sources_searched": total_sources,
        }

    def _get_context(self, full_text: str, chunk_text: str, context_chars: int = 200) -> str:
        pos = full_text.find(chunk_text[:50])
        if pos == -1:
            return chunk_text
        start = max(0, pos - context_chars)
        end = min(len(full_text), pos + len(chunk_text) + context_chars)
        context = full_text[start:end]
        if start > 0:
            context = "..." + context
        if end < len(full_text):
            context = context + "..."
        return context
```

**Step 5: Run tests to verify they pass**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/test_search.py -v`
Expected: All 3 tests PASS.

**Step 6: Commit**

```bash
git add src/bereans/search/ tests/test_search.py
git commit -m "feat: search engine with context enrichment and filtering"
```

---

### Task 9: MCP Server

**Files:**
- Create: `src/bereans/server.py`

**Step 1: Write the MCP server**

Create `src/bereans/server.py`:

```python
"""Bereans Study MCP Server — entry point with 6 tools."""

from mcp.server.fastmcp import FastMCP

from bereans.config import SQLITE_PATH, CHROMA_PATH
from bereans.ingestion.pipeline import IngestionPipeline
from bereans.search.engine import SearchEngine
from bereans.storage.sqlite_store import SQLiteStore
from bereans.storage.vector_store import VectorStore

# Initialize stores
sqlite_store = SQLiteStore(SQLITE_PATH)
sqlite_store.initialize()

vector_store = VectorStore(CHROMA_PATH)
vector_store.initialize()

pipeline = IngestionPipeline(sqlite_store=sqlite_store, vector_store=vector_store)
search_engine = SearchEngine(sqlite_store=sqlite_store, vector_store=vector_store)

mcp = FastMCP("Bereans Study")


@mcp.tool()
async def search(
    query: str,
    top_k: int = 10,
    source_type: str | None = None,
    file_format: str | None = None,
) -> dict:
    """Search the knowledge base with a natural language query.

    Performs semantic search across all ingested documents — Bible texts,
    scientific papers, books, and study materials. Returns the most relevant
    passages with surrounding context.

    Args:
        query: Natural language question or search terms.
        top_k: Number of results to return (default 10).
        source_type: Optional filter — 'file', 'url', or 'api'.
        file_format: Optional filter — 'pdf', 'epub', 'docx', 'html', 'txt', etc.
    """
    return await search_engine.search(
        query=query, top_k=top_k, source_type=source_type, file_format=file_format
    )


@mcp.tool()
async def add_document(
    file_path: str,
    title: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Ingest a local file into the knowledge base.

    Parses the file, extracts all text, chunks it, generates embeddings,
    and stores everything. Supports: PDF, EPUB, DOCX, TXT, Markdown,
    HTML, CSV, JSON.

    Args:
        file_path: Absolute path to the file.
        title: Optional title override (otherwise extracted from the file).
        tags: Optional tags for organization.
    """
    return await pipeline.ingest_file(file_path=file_path, title=title, tags=tags)


@mcp.tool()
async def add_url(
    url: str,
    title: str | None = None,
    tags: list[str] | None = None,
) -> dict:
    """Ingest a web page or file URL into the knowledge base.

    Fetches the URL, detects the content type, parses it, and stores
    everything. Works with web pages (HTML) and direct file links
    (PDF, EPUB, etc.).

    Args:
        url: The URL to fetch and ingest.
        title: Optional title override.
        tags: Optional tags for organization.
    """
    return await pipeline.ingest_url(url=url, title=title, tags=tags)


@mcp.tool()
async def list_sources(
    source_type: str | None = None,
    file_format: str | None = None,
    limit: int = 20,
    offset: int = 0,
) -> dict:
    """List documents in the knowledge base.

    Browse what has been ingested, with optional filtering.

    Args:
        source_type: Filter by 'file', 'url', or 'api'.
        file_format: Filter by format — 'pdf', 'epub', 'txt', etc.
        limit: Max results per page (default 20).
        offset: Pagination offset.
    """
    sources = sqlite_store.list_documents(
        source_type=source_type, file_format=file_format, limit=limit, offset=offset
    )
    total = sqlite_store.count_documents(source_type=source_type, file_format=file_format)
    return {"sources": sources, "total_count": total}


@mcp.tool()
async def delete_source(document_id: str) -> dict:
    """Remove a document and all its chunks from the knowledge base.

    Deletes the document from both SQLite and ChromaDB.

    Args:
        document_id: The UUID of the document to delete.
    """
    doc = sqlite_store.get_document(document_id)
    if doc is None:
        return {"status": "error", "message": f"Document not found: {document_id}"}

    vector_deleted = vector_store.delete_by_document_id(document_id)
    sqlite_deleted = sqlite_store.delete_document(document_id)

    return {
        "status": "success",
        "deleted_chunks": sqlite_deleted,
        "message": f"Deleted document '{doc['title']}' and {sqlite_deleted} chunks",
    }


@mcp.tool()
async def get_stats() -> dict:
    """Get an overview of the entire knowledge base.

    Returns total documents, chunks, breakdowns by format and source type,
    storage size, and the last ingestion time.
    """
    stats = sqlite_store.get_stats()
    stats["vector_chunks"] = vector_store.count()
    return stats


if __name__ == "__main__":
    mcp.run()
```

**Step 2: Smoke test the server starts**

Run: `cd "D:/Bereans Study MCP" && python -c "from bereans.server import mcp; print('Server loaded:', mcp.name)"`
Expected: `Server loaded: Bereans Study`

**Step 3: Commit**

```bash
git add src/bereans/server.py
git commit -m "feat: MCP server with 6 tools — search, add_document, add_url, list_sources, delete_source, get_stats"
```

---

### Task 10: Claude Desktop Configuration

**Files:**
- Modify: `%APPDATA%/Claude/claude_desktop_config.json`

**Step 1: Find the Claude Desktop config file**

Run: `ls "$APPDATA/Claude/claude_desktop_config.json" 2>/dev/null || echo "File not found — check path"`

**Step 2: Read existing config**

Read the file to see if there are existing MCP servers configured.

**Step 3: Add the bereans-study entry**

Add to the `mcpServers` key:

```json
{
  "bereans-study": {
    "command": "python",
    "args": ["-m", "bereans.server"],
    "cwd": "D:/Bereans Study MCP"
  }
}
```

Merge with any existing entries — do not overwrite other servers.

**Step 4: Commit project .gitignore and final state**

```bash
git add -A
git commit -m "chore: final project setup"
git push origin main
```

---

### Task 11: Install Ollama & Pull Model

**Step 1: Verify or install Ollama**

Check if Ollama is installed. If not, guide the user to download from https://ollama.com/download

**Step 2: Pull the embedding model**

Run: `ollama pull nomic-embed-text`
Expected: Model downloads successfully.

**Step 3: Verify Ollama is serving**

Run: `curl http://localhost:11434`
Expected: `Ollama is running`

---

### Task 12: End-to-End Verification

**Step 1: Run full test suite**

Run: `cd "D:/Bereans Study MCP" && python -m pytest tests/ -v`
Expected: All tests pass.

**Step 2: Manual smoke test — ingest a file**

Create a test file and ingest it:

```bash
echo "In the beginning God created the heavens and the earth." > /tmp/genesis1.txt
```

Then via Claude Desktop or direct:

```python
import asyncio
from bereans.ingestion.pipeline import IngestionPipeline
from bereans.storage.sqlite_store import SQLiteStore
from bereans.storage.vector_store import VectorStore
from bereans.config import SQLITE_PATH, CHROMA_PATH

sqlite = SQLiteStore(SQLITE_PATH)
sqlite.initialize()
vector = VectorStore(CHROMA_PATH)
vector.initialize()
pipeline = IngestionPipeline(sqlite, vector)

result = asyncio.run(pipeline.ingest_file("/tmp/genesis1.txt"))
print(result)
```

Expected: `{'status': 'success', 'document_id': '...', 'chunks_created': 1, ...}`

**Step 3: Manual smoke test — search**

```python
from bereans.search.engine import SearchEngine
engine = SearchEngine(sqlite, vector)
results = asyncio.run(engine.search("creation of the world"))
print(results)
```

Expected: Returns the Genesis passage with context.

**Step 4: Final commit and push**

```bash
git add -A
git commit -m "feat: end-to-end verified Bereans Study MCP server"
git push origin main
```
