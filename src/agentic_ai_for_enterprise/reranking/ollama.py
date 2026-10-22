"""Score candidates with the configured BGE text reranker via llama.cpp."""

from __future__ import annotations

import math
from collections.abc import Sequence
from json import JSONDecodeError, dumps, loads
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ..config import Settings
from ..contracts import RetrievedPassage


class OllamaReranker:
    def __init__(self, settings: Settings) -> None:
        self.endpoint = f"{settings.reranker_base_url.rstrip('/')}/v1/rerank"
        self.model = settings.reranker_model
        self.timeout = settings.request_timeout_seconds

    def score(self, query: str, passage: RetrievedPassage) -> float:
        return self._score_many(query, [passage])[0]

    def _score_many(self, query: str, passages: Sequence[RetrievedPassage]) -> list[float]:
        if not passages:
            return []
        body = dumps(
            {
                "model": self.model,
                "query": query,
                "documents": [passage.text for passage in passages],
                "top_n": len(passages),
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                payload = loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                f"Reranker request to {self.endpoint} failed with HTTP {exc.code}: {detail or exc.reason}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(f"Could not reach reranker endpoint {self.endpoint}: {exc.reason}") from exc
        except TimeoutError as exc:
            raise RuntimeError(f"Reranker request to {self.endpoint} timed out after {self.timeout} seconds.") from exc
        except (JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Reranker endpoint {self.endpoint} returned invalid JSON: {exc}") from exc

        if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
            raise RuntimeError(f"Reranker endpoint {self.endpoint} returned no results array.")
        results = payload["results"]
        if len(results) != len(passages):
            raise RuntimeError(
                f"Reranker endpoint {self.endpoint} returned {len(results)} results for {len(passages)} passages."
            )

        scores: dict[int, float] = {}
        for result in results:
            if not isinstance(result, dict):
                raise RuntimeError(f"Reranker endpoint {self.endpoint} returned a malformed result entry.")
            index = result.get("index")
            score = result.get("relevance_score")
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(passages):
                raise RuntimeError(f"Reranker endpoint {self.endpoint} returned an invalid result index: {index!r}.")
            if index in scores:
                raise RuntimeError(f"Reranker endpoint {self.endpoint} returned duplicate result index {index}.")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise RuntimeError(f"Reranker endpoint {self.endpoint} returned an invalid relevance score for index {index}.")
            scores[index] = float(score)

        if len(scores) != len(passages):
            raise RuntimeError(f"Reranker endpoint {self.endpoint} omitted one or more passage scores.")
        return [scores[index] for index in range(len(passages))]


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
    batch_score = getattr(reranker, "_score_many", None)
    scores = (
        batch_score(query, prefix)
        if callable(batch_score)
        else [reranker.score(query, item) for item in prefix]
    )
    if len(scores) != len(prefix):
        raise RuntimeError(f"Reranker returned {len(scores)} scores for {len(prefix)} passages.")
    scored = [(index, item, score) for index, (item, score) in enumerate(zip(prefix, scores, strict=True))]
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
            score_type="bge_reranker_relevance",
        )
        for _, item, score in scored
    ]
    return reranked + ordered[reranker_depth:]
