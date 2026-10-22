"""Hybrid BM25 and vector retrieval with reciprocal-rank fusion."""

from __future__ import annotations

from dataclasses import replace
from collections.abc import Sequence

from ..config import PipelineConfig, Settings
from ..contracts import RetrievedPassage, Retriever
from .bm25 import SQLiteBM25Store
from .vector import PineconeRetriever


def reciprocal_rank_fusion(
    vector_results: Sequence[RetrievedPassage],
    bm25_results: Sequence[RetrievedPassage],
    *,
    fusion_constant: int,
) -> list[RetrievedPassage]:
    """Fuse ranked sources, deduplicate by chunk ID, and preserve source order on ties."""
    if fusion_constant < 1:
        raise ValueError("fusion_constant must be at least 1")

    passages: dict[str, RetrievedPassage] = {}
    scores: dict[str, float] = {}
    stable_order: dict[str, int] = {}
    next_order = 0

    for candidates in (vector_results, bm25_results):
        seen_in_source: set[str] = set()
        rank = 0
        for passage in candidates:
            if passage.chunk_id in seen_in_source:
                continue
            seen_in_source.add(passage.chunk_id)
            rank += 1
            if passage.chunk_id not in passages:
                passages[passage.chunk_id] = passage
                scores[passage.chunk_id] = 0.0
                stable_order[passage.chunk_id] = next_order
                next_order += 1
            scores[passage.chunk_id] += 1.0 / (fusion_constant + rank)

    ordered_ids = sorted(passages, key=lambda chunk_id: (-scores[chunk_id], stable_order[chunk_id]))
    return [
        replace(passages[chunk_id], score=scores[chunk_id], score_type="rrf_score")
        for chunk_id in ordered_ids
    ]


class HybridRetriever:
    """Retrieve from Pinecone and SQLite FTS5, then fuse their ranked candidates."""

    def __init__(self, settings: Settings, pipeline: PipelineConfig) -> None:
        self.vector_store: Retriever = PineconeRetriever(settings)
        self.bm25_store = SQLiteBM25Store(pipeline.bm25_index_path)
        self.fusion_constant = pipeline.fusion_constant

    def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")
        vector_results = self.vector_store.retrieve(query, limit)
        bm25_results = self.bm25_store.retrieve(query, limit)
        return reciprocal_rank_fusion(
            vector_results,
            bm25_results,
            fusion_constant=self.fusion_constant,
        )

    def close(self) -> None:
        for store in (self.vector_store, self.bm25_store):
            close = getattr(store, "close", None)
            if callable(close):
                close()
