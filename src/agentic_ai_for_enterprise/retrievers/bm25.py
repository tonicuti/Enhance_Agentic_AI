"""Local SQLite FTS5 BM25 indexing and retrieval for corpus passages."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

from ..contracts import RetrievedPassage, TextChunk


_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT NOT NULL,
    text TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    published_at TEXT NOT NULL,
    chunk_index INTEGER NOT NULL
)
"""

_CREATE_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    title,
    text,
    content='chunks',
    content_rowid='rowid',
    tokenize='unicode61'
)
"""

_TRIGGERS = (
    """CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
        INSERT INTO chunks_fts(rowid, title, text) VALUES (new.rowid, new.title, new.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, text)
        VALUES ('delete', old.rowid, old.title, old.text);
    END""",
    """CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
        INSERT INTO chunks_fts(chunks_fts, rowid, title, text)
        VALUES ('delete', old.rowid, old.title, old.text);
        INSERT INTO chunks_fts(rowid, title, text) VALUES (new.rowid, new.title, new.text);
    END""",
)


class SQLiteBM25Store:
    """Persist corpus chunks in a local FTS5 index and return ranked passages."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def index(self, chunks: list[TextChunk]) -> int:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            self._ensure_schema(connection)
            with connection:
                connection.execute("DELETE FROM chunks")
                connection.executemany(
                    """INSERT INTO chunks (
                        chunk_id, document_id, title, url, text, source,
                        category, published_at, chunk_index
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (
                            chunk.chunk_id,
                            chunk.document_id,
                            chunk.title,
                            chunk.url,
                            chunk.text,
                            chunk.source or "",
                            chunk.category or "",
                            chunk.published_at or "",
                            chunk.chunk_index,
                        )
                        for chunk in chunks
                    ],
                )
                connection.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
            return len(chunks)
        finally:
            connection.close()

    def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if not self.path.is_file():
            raise RuntimeError(f"BM25 index '{self.path}' does not exist. Run indexing with --backend hybrid first.")
        match_query = _match_query(query)

        connection = self._connect()
        try:
            if not self._has_schema(connection):
                raise RuntimeError(
                    f"BM25 index '{self.path}' is missing its FTS5 schema. Run indexing with --backend hybrid first."
                )
            if not match_query:
                return []
            rows = connection.execute(
                """SELECT c.*, bm25(chunks_fts, 2.0, 1.0) AS bm25_rank
                FROM chunks_fts JOIN chunks AS c ON c.rowid = chunks_fts.rowid
                WHERE chunks_fts MATCH ?
                ORDER BY bm25_rank ASC, c.chunk_id ASC
                LIMIT ?""",
                (match_query, limit),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" in str(exc).lower():
                raise RuntimeError(
                    "This Python SQLite build does not include FTS5 support; install Python with SQLite FTS5 enabled."
                ) from exc
            raise
        finally:
            connection.close()

        return [
            RetrievedPassage(
                document_id=str(row["document_id"]),
                chunk_id=str(row["chunk_id"]),
                title=str(row["title"]),
                url=str(row["url"]),
                text=str(row["text"]),
                score=-float(row["bm25_rank"]),
                source=_optional_string(row["source"]),
                category=_optional_string(row["category"]),
                chunk_index=int(row["chunk_index"]),
                score_type="sqlite_bm25",
            )
            for row in rows
        ]

    def close(self) -> None:
        """Connections are scoped to individual operations, so nothing remains open."""

    def _connect(self) -> sqlite3.Connection:
        try:
            connection = sqlite3.connect(self.path)
            connection.row_factory = sqlite3.Row
            return connection
        except sqlite3.Error as exc:
            raise RuntimeError(f"Could not open BM25 index '{self.path}': {exc}") from exc

    @staticmethod
    def _ensure_schema(connection: sqlite3.Connection) -> None:
        try:
            connection.execute(_CREATE_CHUNKS)
            connection.execute(_CREATE_FTS)
            for statement in _TRIGGERS:
                connection.execute(statement)
        except sqlite3.OperationalError as exc:
            if "no such module: fts5" in str(exc).lower():
                raise RuntimeError(
                    "This Python SQLite build does not include FTS5 support; install Python with SQLite FTS5 enabled."
                ) from exc
            raise

    @staticmethod
    def _has_schema(connection: sqlite3.Connection) -> bool:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name IN ('chunks', 'chunks_fts')"
        ).fetchall()
        return len(rows) == 2


def _match_query(query: str) -> str:
    tokens = re.findall(r"\w+", query, flags=re.UNICODE)
    return " OR ".join(f'"{token}"' for token in tokens)


def _optional_string(value: object) -> str | None:
    return str(value) if value not in (None, "") else None
