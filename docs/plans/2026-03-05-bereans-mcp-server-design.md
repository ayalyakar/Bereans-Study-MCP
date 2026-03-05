# Bereans Study MCP Server — Design Document

**Date:** 2026-03-05
**Status:** Approved

## Purpose

A Python MCP server that acts as a massive, living knowledge base for Bible texts, scientific literature, books, and study materials. Ingests content from files, URLs, and APIs. Stores every detail. Provides semantic search via Claude Desktop.

## Architecture

```
Claude Desktop
    │ MCP Protocol (stdio)
    ▼
MCP Server (Python)
    ├── MCP Tools (6 tools)
    ├── Ingestion Pipeline (parse → chunk → embed → store)
    ├── Search Engine (query → embed → search → enrich)
    └── Storage Layer
        ├── SQLite (documents, metadata, full text, chunk records)
        └── ChromaDB (chunks + embeddings, semantic search)
            │
        Ollama (nomic-embed-text) — local embedding generation
```

### Data Flow: Ingestion

File/URL → Parser (format-specific) → Full text → SQLite → Chunker → Chunks → Ollama embedding → ChromaDB

### Data Flow: Search

Query → Ollama embedding → ChromaDB similarity search → Matching chunks → Enrich from SQLite → Return to Claude

## Data Model

### SQLite Schema

```sql
documents (
    id              TEXT PRIMARY KEY,    -- UUID
    title           TEXT,
    source_type     TEXT NOT NULL,       -- 'file', 'url', 'api'
    source_path     TEXT,
    file_format     TEXT,                -- 'pdf', 'epub', 'csv', etc.
    content_full    TEXT,                -- complete extracted text
    content_hash    TEXT,                -- SHA-256 for deduplication
    metadata        JSON,                -- author, date, tags, etc.
    chunk_count     INTEGER DEFAULT 0,
    created_at      TIMESTAMP,
    updated_at      TIMESTAMP
)

chunks (
    id              TEXT PRIMARY KEY,    -- UUID (same ID in ChromaDB)
    document_id     TEXT NOT NULL,       -- FK → documents
    chunk_index     INTEGER,
    content         TEXT,
    start_char      INTEGER,
    end_char        INTEGER,
    metadata        JSON,                -- heading, page number, etc.
    FOREIGN KEY (document_id) REFERENCES documents(id)
)
```

### ChromaDB

One collection: `knowledge_base`
- **ID**: matches `chunks.id` in SQLite
- **Document**: chunk text
- **Embedding**: from Ollama nomic-embed-text
- **Metadata**: document_id, source_type, file_format, chunk_index

### Chunking Strategy

- ~500 tokens per chunk, ~50 token overlap
- Split on paragraph/sentence boundaries, never mid-sentence
- Hierarchy-aware: preserve heading context in chunk metadata
- Documents < 500 tokens stored as single chunk

## Ingestion Pipeline

### Parser Registry

| Format | Library | Notes |
|--------|---------|-------|
| PDF | pymupdf (fitz) | Preserves page numbers |
| EPUB | ebooklib | Chapter-aware |
| MOBI | mobi + EPUB fallback | Convert then parse |
| DOCX | python-docx | Paragraphs, headings, tables |
| TXT / Markdown | Built-in | Heading-aware splitting |
| HTML | beautifulsoup4 | Strips tags, preserves structure |
| CSV | pandas | Row/group → chunk |
| JSON | Built-in json | Flattens nested structures |

### Ingestion Steps

1. RECEIVE — file path or URL
2. DETECT — format via extension + magic bytes
3. DEDUPLICATE — SHA-256 hash check
4. PARSE — format-specific extraction
5. STORE DOC — full text + metadata → SQLite
6. CHUNK — smart splitting (500 tokens, paragraph boundaries)
7. STORE CHUNKS — chunk records → SQLite
8. EMBED — each chunk → Ollama → vector
9. INDEX — chunks + embeddings → ChromaDB

Atomic: if any step fails, full rollback.

## MCP Tools

### search
- Input: query (str), top_k (int=10), source_type? file_format?
- Output: results with chunk_text, document_title, source_path, relevance_score, context

### add_document
- Input: file_path (str), title?, tags?
- Output: document_id, chunks_created, format_detected, status

### add_url
- Input: url (str), title?, tags?
- Output: document_id, chunks_created, format_detected, status

### list_sources
- Input: source_type?, file_format?, limit (int=20), offset (int=0)
- Output: sources list, total_count

### delete_source
- Input: document_id (str)
- Output: deleted_chunks, status

### get_stats
- Input: none
- Output: total_documents, total_chunks, by_format, by_source_type, storage_size_mb, last_ingested

## Project Structure

```
bereans-study-mcp/
├── pyproject.toml
├── README.md
├── .env.example
├── src/
│   └── bereans/
│       ├── __init__.py
│       ├── server.py           — MCP entry point, tool registration
│       ├── config.py           — settings, paths, constants
│       ├── storage/
│       │   ├── __init__.py
│       │   ├── sqlite_store.py
│       │   └── vector_store.py
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── pipeline.py
│       │   ├── chunker.py
│       │   └── parsers/
│       │       ├── __init__.py
│       │       ├── pdf.py
│       │       ├── epub.py
│       │       ├── docx.py
│       │       ├── html.py
│       │       ├── markdown.py
│       │       ├── csv_json.py
│       │       └── plaintext.py
│       ├── search/
│       │   ├── __init__.py
│       │   └── engine.py
│       └── embeddings/
│           ├── __init__.py
│           └── ollama.py
├── data/                       — runtime, gitignored
│   ├── bereans.db
│   └── chroma/
└── tests/
    ├── test_ingestion.py
    ├── test_search.py
    └── test_storage.py
```

## Error Handling

| Scenario | Behavior |
|----------|----------|
| Ollama not running | Clear error: "Start Ollama with `ollama serve`" |
| Unsupported format | Error with supported formats list |
| Duplicate document | Return existing document ID |
| File not found | Error with path tried |
| URL unreachable | Retry once, then error with status |
| Midway failure | Atomic rollback |
| Empty file | Skip with warning |

## Configuration

### .env
```
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=nomic-embed-text
SQLITE_PATH=./data/bereans.db
CHROMA_PATH=./data/chroma
CHUNK_SIZE=500
CHUNK_OVERLAP=50
```

### Claude Desktop (claude_desktop_config.json)
```json
{
  "mcpServers": {
    "bereans-study": {
      "command": "python",
      "args": ["-m", "bereans.server"],
      "cwd": "D:/Bereans Study MCP"
    }
  }
}
```

## Key Design Decisions

1. **SQLite + ChromaDB dual storage** — each DB does what it's best at. SQLite for complete data preservation and SQL queries. ChromaDB for fast vector search.
2. **Chunk ID mirroring** — same UUID in both stores eliminates mapping overhead.
3. **Ollama for embeddings** — local, free, hot-swappable models.
4. **Modular parsers** — add new formats by dropping a file in parsers/.
5. **Atomic ingestion** — no ghost data from partial failures.
6. **Context enrichment** — search returns surrounding text, not just isolated chunks.
