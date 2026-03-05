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
