"""Ingestion pipeline — orchestrates parsing, chunking, embedding, and storage."""

import hashlib
from pathlib import Path

import httpx

from bereans.config import OLLAMA_HOST, OLLAMA_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
from bereans.embeddings.ollama import OllamaEmbedder
from bereans.ingestion.chunker import TextChunker
from bereans.ingestion.parsers import get_parser
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
        elif url.lower().endswith(".csv"):
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
            # Rollback on failure — clean up both stores
            self.sqlite.delete_document(doc_id)
            try:
                self.vector.delete_by_document_id(doc_id)
            except Exception:
                pass  # Best-effort vector cleanup
            return {"status": "error", "message": f"Ingestion failed (rolled back): {e}"}
