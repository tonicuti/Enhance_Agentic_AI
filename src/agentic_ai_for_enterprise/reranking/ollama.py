"""Score candidates with the configured Qwen3 text reranker via Ollama chat."""

from __future__ import annotations

import math
import re
from collections.abc import Sequence

from langchain_ollama import ChatOllama

from ..config import Settings
from ..contracts import RetrievedPassage


RERANK_INSTRUCTION = "Given a user question, determine whether the passage answers it."
RERANK_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and Instruct. '
    'The answer must be only "yes" or "no".'
)


class OllamaReranker:
    def __init__(self, settings: Settings) -> None:
        self.llm = ChatOllama(
            model=settings.reranker_model,
            base_url=settings.ollama_base_url,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            num_predict=1,
            timeout=settings.request_timeout_seconds,
        )

    def score(self, query: str, passage: RetrievedPassage) -> float:
        message = self.llm.invoke(
            [
                ("system", RERANK_SYSTEM),
                (
                    "human",
                    f"<Instruct>: {RERANK_INSTRUCTION}\n\n"
                    f"<Query>: {query}\n\n<Document>: {passage.text}",
                ),
            ]
        )
        # ChatOllama exposes token log probabilities through response metadata.
        # API reference: https://docs.langchain.com/oss/python/integrations/chat/ollama#log-probabilities
        probabilities = _yes_no_probabilities(message.response_metadata.get("logprobs"))
        if probabilities is not None:
            yes_logprob, no_logprob = probabilities
            max_logprob = max(yes_logprob, no_logprob)
            yes_score = math.exp(yes_logprob - max_logprob)
            no_score = math.exp(no_logprob - max_logprob)
            return yes_score / (yes_score + no_score)
        decision = _yes_no_decision(str(message.content))
        if decision is None:
            raise RuntimeError("The configured Ollama reranker returned neither a yes/no score nor a yes/no answer.")
        return 1.0 if decision else 0.0


def rerank(
    query: str,
    candidates: Sequence[RetrievedPassage],
    *,
    enabled: bool,
    settings: Settings,
    reranker_depth: int,
    ranker: OllamaReranker | None = None,
) -> list[RetrievedPassage]:
    """Rerank only the configured prefix; disabled mode preserves original ordering."""
    ordered = list(candidates)
    if not enabled or not ordered:
        return ordered
    if reranker_depth < 1:
        raise ValueError("reranker_depth must be at least 1")
    reranker = ranker or OllamaReranker(settings)
    prefix = ordered[:reranker_depth]
    scored = [(index, item, reranker.score(query, item)) for index, item in enumerate(prefix)]
    scored.sort(key=lambda result: (-result[2], result[0]))
    reranked = [
        RetrievedPassage(
            document_id=item.document_id,
            chunk_id=item.chunk_id,
            title=item.title,
            url=item.url,
            text=item.text,
            score=score,
            source=item.source,
            category=item.category,
            chunk_index=item.chunk_index,
            score_type="qwen3_reranker_yes_probability",
        )
        for _, item, score in scored
    ]
    return reranked + ordered[reranker_depth:]


def _yes_no_probabilities(logprobs: object) -> tuple[float, float] | None:
    if not isinstance(logprobs, list) or not logprobs:
        return None
    first = logprobs[0]
    if not isinstance(first, dict):
        return None
    alternatives = first.get("top_logprobs")
    if not isinstance(alternatives, list):
        return None
    scores: dict[str, float] = {}
    for alternative in alternatives:
        if not isinstance(alternative, dict):
            continue
        token = str(alternative.get("token", "")).strip().casefold()
        logprob = alternative.get("logprob")
        if token in {"yes", "no"} and isinstance(logprob, (int, float)):
            scores[token] = float(logprob)
    if "yes" not in scores or "no" not in scores:
        return None
    return scores["yes"], scores["no"]


def _yes_no_decision(content: str) -> bool | None:
    token = re.match(r"\s*(yes|no)\b", content, flags=re.IGNORECASE)
    if token is None:
        return None
    return token.group(1).casefold() == "yes"
