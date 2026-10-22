"""Wire configured retrieval, reranking, response generation, and ReAct."""

from __future__ import annotations

import time
from dataclasses import replace
from typing import TYPE_CHECKING

from .agent import answer_with_react
from .config import DEFAULT_CONFIG_PATH, PipelineConfig, Settings
from .contracts import PipelineResult, RetrievedPassage, RetrievalTrace, Retriever
from .reranking import OllamaReranker, rerank
from .response import generate_answer

if TYPE_CHECKING:
    from .retrievers.graph.neo4j_store import Neo4jGraphStore


class ConfiguredRetriever:
    """Apply the frozen reranker configuration consistently to backend retrieval."""

    def __init__(self, backend: Retriever, *, settings: Settings, pipeline: PipelineConfig) -> None:
        self.backend = backend
        self.settings = settings
        self.pipeline = pipeline
        self.model_call_count = 0
        self._ranker: OllamaReranker | None = None

    def retrieve(
        self,
        query: str,
        limit: int,
        *,
        track_model_calls: bool = True,
    ) -> list[RetrievedPassage]:
        candidates = self.backend.retrieve(query, limit)
        if track_model_calls and getattr(self.backend, "is_graph_backend", False):
            self.model_call_count += 1
        if self.pipeline.reranker_enabled and candidates:
            if track_model_calls:
                self.model_call_count += min(len(candidates), self.pipeline.reranker_depth)
            if self._ranker is None:
                self._ranker = OllamaReranker(self.settings)
        return rerank(
            query,
            candidates,
            enabled=self.pipeline.reranker_enabled,
            settings=self.settings,
            reranker_depth=self.pipeline.reranker_depth,
            ranker=self._ranker,
        )

    def close(self) -> None:
        close = getattr(self.backend, "close", None)
        if callable(close):
            close()


def get_retriever(backend: str, settings: Settings, pipeline: PipelineConfig) -> ConfiguredRetriever:
    if backend == "vector":
        from .retrievers.vector import PineconeRetriever

        base: Retriever = PineconeRetriever(settings)
    elif backend == "graph":
        from .retrievers.graph import Neo4jGraphStore

        base = _GraphRetriever(Neo4jGraphStore(settings), pipeline.graph_traversal_depth)
    elif backend == "hybrid":
        from .retrievers.hybrid import HybridRetriever

        base = HybridRetriever(settings, pipeline)
    else:
        raise ValueError("backend must be 'vector', 'graph', or 'hybrid'")
    return ConfiguredRetriever(base, settings=settings, pipeline=pipeline)


def answer_question(
    question: str,
    backend: str,
    *,
    mode: str = "agentic",
    pipeline: PipelineConfig | None = None,
    settings: Settings | None = None,
) -> PipelineResult:
    if not isinstance(question, str) or not question.strip() or len(question) > 4000:
        raise ValueError("Question must contain between 1 and 4000 characters.")
    settings = settings or Settings.from_env()
    pipeline = pipeline or PipelineConfig.load(DEFAULT_CONFIG_PATH)
    pipeline = replace(pipeline, selected_backend=backend)
    pipeline.validate()
    application = RagApplication(backend=backend, settings=settings, pipeline=pipeline)
    try:
        return application.answer(question, mode=mode)
    finally:
        application.close()


class RagApplication:
    def __init__(self, *, backend: str, settings: Settings, pipeline: PipelineConfig) -> None:
        self.backend = backend
        self.settings = settings
        self.pipeline = replace(pipeline, selected_backend=backend)
        self.pipeline.validate()
        self.retriever = get_retriever(backend, settings, self.pipeline)

    def retrieve(self, question: str) -> RetrievalTrace:
        """Run configured retrieval and reranking without generating an answer."""
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain between 1 and 4000 characters.")
        started = time.perf_counter()
        model_calls_before = self.retriever.model_call_count
        passages = self.retriever.retrieve(question, self.pipeline.candidate_depth)
        return RetrievalTrace(
            passages=tuple(passages),
            retrieval_calls=1,
            model_calls=self.retriever.model_call_count - model_calls_before,
            elapsed_seconds=time.perf_counter() - started,
        )

    def retrieve_passages(
        self,
        question: str,
        *,
        track_model_calls: bool = True,
    ) -> tuple[RetrievedPassage, ...]:
        """Return ranked passages without timing; callers can suppress model-call accounting."""
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain between 1 and 4000 characters.")
        return tuple(
            self.retriever.retrieve(
                question,
                self.pipeline.candidate_depth,
                track_model_calls=track_model_calls,
            )
        )

    def answer(self, question: str, *, mode: str = "agentic") -> PipelineResult:
        if not isinstance(question, str) or not question.strip() or len(question) > 4000:
            raise ValueError("Question must contain between 1 and 4000 characters.")
        started = time.perf_counter()

        model_calls_before = self.retriever.model_call_count
        if mode == "standard":
            passages = self.retriever.retrieve(question, self.pipeline.candidate_depth)
            answer = generate_answer(question, passages, self.settings, self.pipeline)
            generation_calls = 1 if passages else 0
            return PipelineResult(
                answer=answer,
                passages=tuple(passages),
                retrieval_calls=1,
                model_calls=self.retriever.model_call_count - model_calls_before + generation_calls,
                elapsed_seconds=time.perf_counter() - started,
            )
        if mode == "agentic":
            run = answer_with_react(question, self.retriever, self.settings, self.pipeline)
            return PipelineResult(
                answer=run.answer,
                passages=run.passages,
                retrieval_calls=run.retrieval_calls,
                model_calls=run.model_calls + self.retriever.model_call_count - model_calls_before,
                elapsed_seconds=time.perf_counter() - started,
                actions=run.retrieval_queries,
            )
        raise ValueError("mode must be 'standard' or 'agentic'")

    def close(self) -> None:
        self.retriever.close()


class _GraphRetriever:
    is_graph_backend = True

    def __init__(self, store: Neo4jGraphStore, traversal_depth: int) -> None:
        self.store = store
        self.traversal_depth = traversal_depth

    def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]:
        return self.store.retrieve(query, limit, traversal_depth=self.traversal_depth)

    def close(self) -> None:
        self.store.close()
