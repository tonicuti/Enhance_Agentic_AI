"""Load MultiHopRAG questions while isolating gold evidence from indexing code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from ..contracts import BenchmarkQuestion
from ..ingestion.corpus import canonicalize_url


def load_questions(path: Path) -> tuple[list[BenchmarkQuestion], str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read benchmark file at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Benchmark file is not valid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(raw, list):
        raise ValueError("Benchmark JSON root must be an array of question objects.")

    fingerprint = hashlib.sha256(
        json.dumps(raw, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    questions: list[BenchmarkQuestion] = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            raise ValueError(f"Benchmark row {index} must be an object.")
        query = record.get("query")
        answer = record.get("answer")
        question_type = record.get("question_type")
        evidence = record.get("evidence_list")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"Benchmark row {index} has no valid query.")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"Benchmark row {index} has no valid answer.")
        if not isinstance(question_type, str) or not question_type.strip():
            raise ValueError(f"Benchmark row {index} has no valid question_type.")
        if not isinstance(evidence, list):
            raise ValueError(f"Benchmark row {index} has no evidence_list array.")
        urls: list[str] = []
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("url"), str):
                raise ValueError(f"Benchmark row {index} contains an invalid evidence item.")
            try:
                urls.append(canonicalize_url(item["url"]))
            except ValueError as exc:
                raise ValueError(f"Benchmark row {index} contains an invalid evidence URL.") from exc
        digest = hashlib.sha256(f"{index}\0{query}".encode("utf-8")).hexdigest()[:16]
        questions.append(
            BenchmarkQuestion(
                question_id=digest,
                query=query.strip(),
                answer=answer.strip(),
                question_type=question_type.strip(),
                evidence_urls=tuple(dict.fromkeys(urls)),
            )
        )
    return questions, fingerprint
