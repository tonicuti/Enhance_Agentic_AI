"""Ranking, answer, and citation metrics for benchmark reports."""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass

from ..contracts import BenchmarkQuestion, PipelineResult


@dataclass(frozen=True, slots=True)
class QuestionMetrics:
    evidence_recall_at_5: float
    evidence_recall_at_10: float
    ndcg_at_10: float
    answer_exact_match: float
    answer_token_f1: float
    citation_precision: float
    citation_recall: float
    latency_seconds: float
    retrieval_calls: int
    model_calls: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def score_run(question: BenchmarkQuestion, result: PipelineResult) -> QuestionMetrics:
    gold_urls = set(question.evidence_urls)
    retrieved_urls = _unique_urls([passage.url for passage in result.passages])
    cited_urls = _unique_urls(result.answer.sources)
    return QuestionMetrics(
        evidence_recall_at_5=_recall(retrieved_urls[:5], gold_urls),
        evidence_recall_at_10=_recall(retrieved_urls[:10], gold_urls),
        ndcg_at_10=_ndcg(retrieved_urls[:10], gold_urls),
        answer_exact_match=float(normalize_answer(result.answer.context) == normalize_answer(question.answer)),
        answer_token_f1=token_f1(result.answer.context, question.answer),
        citation_precision=(len(set(cited_urls) & gold_urls) / len(set(cited_urls))) if cited_urls else 0.0,
        citation_recall=_recall(cited_urls, gold_urls),
        latency_seconds=result.elapsed_seconds,
        retrieval_calls=result.retrieval_calls,
        model_calls=result.model_calls,
    )


def normalize_answer(value: str) -> str:
    normalized = value.casefold().translate(str.maketrans("", "", string.punctuation))
    words = [word for word in normalized.split() if word not in {"a", "an", "the"}]
    return " ".join(words)


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted or not expected:
        return float(predicted == expected)
    overlap = sum((Counter(predicted) & Counter(expected)).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall)


def _unique_urls(urls: list[str]) -> list[str]:
    return list(dict.fromkeys(urls))


def _recall(retrieved: list[str], relevant: set[str]) -> float:
    return len(set(retrieved) & relevant) / len(relevant) if relevant else 0.0


def _ndcg(retrieved: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    dcg = sum((1.0 / math.log2(rank + 2)) for rank, url in enumerate(retrieved) if url in relevant)
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(relevant), 10)))
    return dcg / ideal if ideal else 0.0
