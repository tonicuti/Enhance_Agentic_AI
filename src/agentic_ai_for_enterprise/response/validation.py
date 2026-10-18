"""Enforce the response contract and allowlist citations to retrieved URLs."""

from __future__ import annotations

from ..contracts import RagAnswer
from ..ingestion.corpus import canonicalize_url


INSUFFICIENT_EVIDENCE_TEXT = "The indexed documents do not provide enough evidence to answer this question."


def insufficient_evidence() -> RagAnswer:
    return RagAnswer(role="assistant", context=INSUFFICIENT_EVIDENCE_TEXT, sources=[])


def validate_answer(answer: RagAnswer, evidence_urls: list[str] | tuple[str, ...]) -> RagAnswer:
    if answer.role != "assistant" or not answer.context.strip():
        return insufficient_evidence()
    try:
        allowed = {canonicalize_url(url) for url in evidence_urls}
        proposed = [canonicalize_url(url) for url in answer.sources]
    except (TypeError, ValueError):
        return insufficient_evidence()
    sources = list(dict.fromkeys(url for url in proposed if url in allowed))
    if len(sources) != len(set(answer.sources)) or not sources:
        return insufficient_evidence()
    return RagAnswer(role="assistant", context=answer.context.strip(), sources=sources)
