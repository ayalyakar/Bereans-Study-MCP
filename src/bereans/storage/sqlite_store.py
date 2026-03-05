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
                 chunk["start_char"], chunk["end_char"],
                 json.dumps(chunk.get("metadata", {}))),
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
        self, source_type=None, file_format=None, limit=20, offset=0,
    ) -> list[dict]:
        query = ("SELECT id, title, source_type, source_path, file_format, "
                 "chunk_count, created_at FROM documents")
        conditions: list[str] = []
        params: list = []
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

    def count_documents(self, source_type=None, file_format=None) -> int:
        query = "SELECT COUNT(*) FROM documents"
        conditions: list[str] = []
        params: list = []
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
        total_docs = self._conn.execute(
            "SELECT COUNT(*) FROM documents"
        ).fetchone()[0]
        total_chunks = self._conn.execute(
            "SELECT COUNT(*) FROM chunks"
        ).fetchone()[0]
        by_format = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT file_format, COUNT(*) FROM documents GROUP BY file_format"
            ).fetchall()
        }
        by_source = {
            r[0]: r[1]
            for r in self._conn.execute(
                "SELECT source_type, COUNT(*) FROM documents GROUP BY source_type"
            ).fetchall()
        }
        last = self._conn.execute(
            "SELECT created_at FROM documents ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        db_size = (
            self.db_path.stat().st_size / (1024 * 1024)
            if self.db_path.exists()
            else 0
        )
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
