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
