"""Parameter-safe Neo4j graph persistence and bounded provenance retrieval."""

from __future__ import annotations

import logging
import re
from collections import defaultdict

from neo4j import GraphDatabase, Query

from ...config import Settings
from ...contracts import RetrievedPassage, TextChunk
from .extraction import GraphExtractor
from .schema import GraphRecord, entity_id, normalize_entity_name, relation_id

# All Cypher values are passed as parameters; see
# https://neo4j.com/docs/python-manual/current/query-simple/.

LOGGER = logging.getLogger(__name__)


class Neo4jGraphStore:
    def __init__(self, settings: Settings) -> None:
        uri, username, password = settings.require_graph()
        self.settings = settings
        self.driver = GraphDatabase.driver(
            uri,
            auth=(username, password),
            connection_timeout=settings.request_timeout_seconds,
            max_transaction_retry_time=min(settings.request_timeout_seconds, 30),
        )
        self._verified = False

    def close(self) -> None:
        self.driver.close()

    def ensure_schema(self) -> None:
        self._verify_connectivity()
        self._execute_query(
            "CREATE CONSTRAINT article_id IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE",
            database_=self.settings.neo4j_database,
        )
        self._execute_query(
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            database_=self.settings.neo4j_database,
        )
        self._execute_query(
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            database_=self.settings.neo4j_database,
        )
        self._execute_query(
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name_normalized)",
            database_=self.settings.neo4j_database,
        )
        self._execute_query(
            "CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS FOR (e:Entity) ON EACH [e.name]",
            database_=self.settings.neo4j_database,
        )

    def index(self, chunks: list[TextChunk], records: list[GraphRecord]) -> int:
        if not chunks:
            return 0
        record_by_chunk = {record.chunk_id: record for record in records}
        missing = [chunk.chunk_id for chunk in chunks if chunk.chunk_id not in record_by_chunk]
        if missing:
            raise ValueError(f"Graph extraction is missing {len(missing)} chunk record(s).")
        self.ensure_schema()
        parameters = [_chunk_row(chunk) for chunk in chunks]
        self._execute_query(
            """
            UNWIND $rows AS row
            MERGE (a:Article {id: row.document_id})
            SET a.title = row.title, a.url = row.url, a.source = row.source,
                a.category = row.category, a.published_at = row.published_at
            MERGE (c:Chunk {id: row.chunk_id})
            SET c.text = row.text, c.chunk_index = row.chunk_index,
                c.document_id = row.document_id, c.url = row.url,
                c.title = row.title, c.source = row.source, c.category = row.category
            MERGE (a)-[:HAS_CHUNK]->(c)
            """,
            rows=parameters,
            database_=self.settings.neo4j_database,
        )

        entity_rows: dict[str, dict[str, str]] = {}
        mention_rows: list[dict[str, str]] = []
        relation_rows: list[dict[str, object]] = []
        chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        for record in records:
            chunk = chunk_by_id.get(record.chunk_id)
            if chunk is None:
                LOGGER.warning("Skipping extracted graph data for unknown chunk %s.", record.chunk_id)
                continue
            name_to_id: dict[str, str] = {}
            for entity in record.entities:
                eid = entity_id(entity.name, entity.entity_type)
                normalized = normalize_entity_name(entity.name)
                name_to_id[normalized] = eid
                entity_rows[eid] = {"id": eid, "name": entity.name, "name_normalized": normalized, "entity_type": entity.entity_type}
                mention_rows.append({"id": f"{chunk.chunk_id}:{eid}", "chunk_id": chunk.chunk_id, "entity_id": eid})
            for relation in record.relations:
                subject_id = name_to_id.get(normalize_entity_name(relation.subject))
                object_id = name_to_id.get(normalize_entity_name(relation.object))
                if not subject_id or not object_id:
                    continue
                relation_rows.append(
                    {
                        "id": relation_id(record.chunk_id, relation),
                        "subject_id": subject_id,
                        "object_id": object_id,
                        "predicate": relation.predicate,
                        "confidence": relation.confidence,
                        "chunk_id": record.chunk_id,
                        "url": chunk.url,
                    }
                )
        if entity_rows:
            self._execute_query(
                """
                UNWIND $rows AS row
                MERGE (e:Entity {id: row.id})
                SET e.name = row.name, e.name_normalized = row.name_normalized,
                    e.entity_type = row.entity_type
                """,
                rows=list(entity_rows.values()),
                database_=self.settings.neo4j_database,
            )
        if mention_rows:
            self._execute_query(
                """
                UNWIND $rows AS row
                MATCH (c:Chunk {id: row.chunk_id})
                MATCH (e:Entity {id: row.entity_id})
                MERGE (c)-[:MENTIONS {id: row.id}]->(e)
                """,
                rows=mention_rows,
                database_=self.settings.neo4j_database,
            )
        if relation_rows:
            self._execute_query(
                """
                UNWIND $rows AS row
                MATCH (s:Entity {id: row.subject_id})
                MATCH (o:Entity {id: row.object_id})
                MERGE (s)-[r:RELATED {id: row.id}]->(o)
                SET r.predicate = row.predicate, r.confidence = row.confidence,
                    r.chunk_id = row.chunk_id, r.url = row.url
                """,
                rows=relation_rows,
                database_=self.settings.neo4j_database,
            )
        return len(chunks)

    def retrieve(self, query: str, limit: int, traversal_depth: int = 2) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if limit < 1 or not 1 <= traversal_depth <= 4:
            raise ValueError("limit must be positive and traversal_depth must be between 1 and 4")
        entities = GraphExtractor(self.settings).extract_query_entities(query)
        terms = sorted(
            {
                _escape_fulltext_term(normalize_entity_name(item.name))
                for item in entities
                if item.name.strip()
            }
        )
        terms = [term for term in terms if term]
        if not terms:
            terms = _query_terms(query)
        if not terms:
            return []

        records, _, _ = self._execute_query(
            """
            UNWIND $terms AS term
            CALL db.index.fulltext.queryNodes('entity_name_fulltext', term)
            YIELD node AS seed, score AS lexical_score
            WITH DISTINCT seed, term, lexical_score
            WITH seed, term, lexical_score ORDER BY lexical_score DESC LIMIT 100
            MATCH p=(seed)-[:RELATED*0..4]-(neighbor:Entity)
            WHERE length(p) <= $traversal_depth
            MATCH (c:Chunk)-[:MENTIONS]->(neighbor)
            MATCH (a:Article)-[:HAS_CHUNK]->(c)
            WITH c, a, seed, neighbor, min(length(p)) AS hops,
                 count(DISTINCT term) AS term_hits,
                 collect(DISTINCT seed.name) AS matched_entities,
                 max(reduce(confidence = 1.0, rel IN relationships(p) |
                     confidence * coalesce(rel.confidence, 1.0))) AS relation_confidence,
                 max(lexical_score) AS lexical_score
            WITH c, a, max(term_hits * 2.0 + (1.0 / (1.0 + hops)) +
                 relation_confidence + lexical_score) AS score,
                 matched_entities
            RETURN c.id AS chunk_id, c.document_id AS document_id,
                   c.title AS title, c.url AS url, c.text AS text,
                   c.chunk_index AS chunk_index, c.source AS source,
                   c.category AS category, score,
                   matched_entities
            ORDER BY score DESC, chunk_id ASC
            LIMIT $limit
            """,
            terms=terms,
            traversal_depth=traversal_depth,
            limit=limit,
            database_=self.settings.neo4j_database,
            routing_="r",
        )
        if not records:
            fallback_terms = _query_terms(query)
            if fallback_terms != terms:
                records, _, _ = self._search_terms(
                    fallback_terms, limit=limit, traversal_depth=traversal_depth
                )
        return [
            RetrievedPassage(
                document_id=str(row["document_id"]),
                chunk_id=str(row["chunk_id"]),
                title=str(row["title"]),
                url=str(row["url"]),
                text=str(row["text"]),
                score=float(row["score"]),
                source=str(row["source"]) if row["source"] else None,
                category=str(row["category"]) if row["category"] else None,
                chunk_index=int(row["chunk_index"]),
                score_type="entity_relation_match",
            )
            for row in records
        ]

    def _verify_connectivity(self) -> None:
        if not self._verified:
            self.driver.verify_connectivity()
            self._verified = True

    def _execute_query(self, query: str, **kwargs: object):
        """Run parameterized Cypher with a server-side transaction timeout."""
        self._verify_connectivity()
        kwargs.setdefault("database_", self.settings.neo4j_database)
        return self.driver.execute_query(
            Query(query, timeout=self.settings.request_timeout_seconds),
            **kwargs,
        )

    def _search_terms(self, terms: list[str], *, limit: int, traversal_depth: int):
        return self._execute_query(
            """
            UNWIND $terms AS term
            CALL db.index.fulltext.queryNodes('entity_name_fulltext', term)
            YIELD node AS seed, score AS lexical_score
            WITH DISTINCT seed, term, lexical_score
            WITH seed, term, lexical_score ORDER BY lexical_score DESC LIMIT 100
            MATCH p=(seed)-[:RELATED*0..4]-(neighbor:Entity)
            WHERE length(p) <= $traversal_depth
            MATCH (c:Chunk)-[:MENTIONS]->(neighbor)
            MATCH (a:Article)-[:HAS_CHUNK]->(c)
            WITH c, a, seed, neighbor, min(length(p)) AS hops,
                 count(DISTINCT term) AS term_hits,
                 collect(DISTINCT seed.name) AS matched_entities,
                 max(reduce(confidence = 1.0, rel IN relationships(p) |
                     confidence * coalesce(rel.confidence, 1.0))) AS relation_confidence,
                 max(lexical_score) AS lexical_score
            WITH c, a, max(term_hits * 2.0 + (1.0 / (1.0 + hops)) +
                 relation_confidence + lexical_score) AS score,
                 matched_entities
            RETURN c.id AS chunk_id, c.document_id AS document_id,
                   c.title AS title, c.url AS url, c.text AS text,
                   c.chunk_index AS chunk_index, c.source AS source,
                   c.category AS category, score,
                   matched_entities
            ORDER BY score DESC, chunk_id ASC
            LIMIT $limit
            """,
            terms=terms,
            traversal_depth=traversal_depth,
            limit=limit,
            routing_="r",
        )


def _chunk_row(chunk: TextChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "url": chunk.url,
        "text": chunk.text,
        "chunk_index": chunk.chunk_index,
        "source": chunk.source or "",
        "category": chunk.category or "",
        "published_at": chunk.published_at or "",
    }


def _query_terms(query: str) -> list[str]:
    terms = list(dict.fromkeys(term.casefold() for term in re.findall(r"[\w'-]{3,}", query)))[:12]
    return [_escape_fulltext_term(term) for term in terms]


def _escape_fulltext_term(term: str) -> str:
    safe = re.sub(r"[^\w\s'-]", " ", term).strip()
    return '"' + safe.replace('"', '\\"') + '"' if safe else ""
