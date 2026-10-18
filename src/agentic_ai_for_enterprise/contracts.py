"""Typed contracts shared by ingestion, retrievers, agents, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True, slots=True)
class Article:
    document_id: str
    title: str
    url: str
    body: str
    source: str | None = None
    category: str | None = None
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class TextChunk:
    chunk_id: str
    document_id: str
    title: str
    url: str
    text: str
    chunk_index: int
    source: str | None = None
    category: str | None = None
    published_at: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedPassage:
    document_id: str
    chunk_id: str
    title: str
    url: str
    text: str
    score: float
    source: str | None = None
    category: str | None = None
    chunk_index: int = 0
    score_type: str = "retrieval"


@dataclass(frozen=True, slots=True)
class RagAnswer:
    role: str
    context: str
    sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, str | list[str]]:
        return {"role": self.role, "context": self.context, "sources": list(self.sources)}


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    question_id: str
    query: str
    answer: str
    question_type: str
    evidence_urls: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    passages: tuple[RetrievedPassage, ...]
    retrieval_calls: int
    model_calls: int
    elapsed_seconds: float
    actions: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PipelineResult:
    answer: RagAnswer
    passages: tuple[RetrievedPassage, ...]
    retrieval_calls: int
    model_calls: int
    elapsed_seconds: float
    actions: tuple[str, ...] = ()


class Retriever(Protocol):
    """Common retrieval interface used by standard and ReAct RAG."""

    def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]: ...
