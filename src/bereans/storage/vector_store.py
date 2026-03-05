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

    def add_chunks(self, ids, documents, embeddings, metadatas):
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(self, query_embedding, top_k=10, where=None):
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def delete_by_document_id(self, document_id):
        existing = self.collection.get(where={"document_id": document_id}, include=[])
        count = len(existing["ids"])
        if count > 0:
            self.collection.delete(ids=existing["ids"])
        return count

    def count(self):
        return self.collection.count()
