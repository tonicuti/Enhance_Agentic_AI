"""Run paired RAG benchmark conditions and persist resumable question records."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..application import RagApplication
from ..config import PipelineConfig, Settings
from ..contracts import BenchmarkQuestion, PipelineResult, RetrievalQualityTrace, RetrievalTrace
from .metrics import (
    QuestionMetrics,
    RetrievalMetrics,
    RetrievalQualityMetrics,
    score_retrieval,
    score_retrieval_quality,
    score_run,
)


LOGGER = logging.getLogger(__name__)


def run_condition(
    *,
    questions: list[BenchmarkQuestion],
    dataset_fingerprint: str,
    split_fingerprint: str,
    phase: str,
    split_name: str,
    backend: str,
    mode: str,
    pipeline: PipelineConfig,
    settings: Settings,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reranker_label = "enabled" if pipeline.reranker_enabled else "disabled"
    filename = f"{phase}-{split_name}-{backend}-{mode}-reranker-{reranker_label}.jsonl"
    result_path = output_dir / filename
    run_key = _run_key(
        dataset_fingerprint=dataset_fingerprint,
        split_fingerprint=split_fingerprint,
        phase=phase,
        split_name=split_name,
        backend=backend,
        mode=mode,
        pipeline=pipeline,
    )
    existing = _read_resume_rows(result_path, run_key)
    completed = {row["question_id"] for row in existing}
    application = RagApplication(backend=backend, settings=settings, pipeline=pipeline)
    try:
        for number, question in enumerate(questions, start=1):
            if question.question_id in completed:
                continue
            LOGGER.info("%s / %s: question %d of %d", backend, mode, number, len(questions))
            if mode == "retrieval_quality":
                passages = application.retrieve_passages(question.query, track_model_calls=False)
                result = RetrievalQualityTrace(passages=passages)
                metrics = score_retrieval_quality(question, passages)
            elif mode == "retrieval":
                result = application.retrieve(question.query)
                metrics = score_retrieval(question, result)
            else:
                result = application.answer(question.query, mode=mode)
                metrics = score_run(question, result)
            row = _question_row(
                question=question,
                result=result,
                metrics=metrics,
                run_key=run_key,
                run_id=run_id,
                phase=phase,
                split_name=split_name,
                backend=backend,
                mode=mode,
                pipeline=pipeline,
                settings=settings,
            )
            with result_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                stream.flush()
            existing.append(row)
            completed.add(question.question_id)
    finally:
        application.close()

    summary = _aggregate(existing, mode=mode)
    summary.update(
        {
            "run_id": run_id,
            "run_key": run_key,
            "phase": phase,
            "split": split_name,
            "backend": backend,
            "mode": mode,
            "reranker_enabled": pipeline.reranker_enabled,
            "configuration": asdict(pipeline),
            "dataset_fingerprint": dataset_fingerprint,
            "split_fingerprint": split_fingerprint,
            "question_results": str(result_path),
        }
    )
    summary_path = result_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def _question_row(
    *,
    question: BenchmarkQuestion,
    result: PipelineResult | RetrievalQualityTrace | RetrievalTrace,
    metrics: QuestionMetrics | RetrievalQualityMetrics | RetrievalMetrics,
    run_key: str,
    run_id: str,
    phase: str,
    split_name: str,
    backend: str,
    mode: str,
    pipeline: PipelineConfig,
    settings: Settings,
) -> dict[str, Any]:
    row = {
        "run_id": run_id,
        "run_key": run_key,
        "question_id": question.question_id,
        "question_type": question.question_type,
        "query": question.query,
        "gold_evidence_urls": list(question.evidence_urls),
        "phase": phase,
        "split": split_name,
        "backend": backend,
        "mode": mode,
        "reranker_enabled": pipeline.reranker_enabled,
        "configuration": asdict(pipeline),
        "models": {
            "embedding": settings.embedding_model,
            "reranker": settings.reranker_model if pipeline.reranker_enabled else None,
        },
        "retrieved_evidence": [
            {
                "rank": rank,
                "chunk_id": passage.chunk_id,
                "document_id": passage.document_id,
                "title": passage.title,
                "url": passage.url,
                "score": passage.score,
                "score_type": passage.score_type,
            }
            for rank, passage in enumerate(result.passages, start=1)
        ],
        "retrieval_actions": list(getattr(result, "actions", ())),
        "metrics": metrics.to_dict(),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    if isinstance(result, PipelineResult):
        row["gold_answer"] = question.answer
        row["models"]["generation"] = settings.generation_model
        row["response"] = result.answer.to_dict()
    if backend == "graph":
        row["models"]["query_extraction"] = settings.generation_model
    return row


def _read_resume_rows(path: Path, run_key: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Cannot resume from malformed result row {line_number} in {path}.") from exc
            if row.get("run_key") != run_key:
                raise ValueError(f"Existing results in {path} belong to another configuration; choose another run ID.")
            rows.append(row)
    return rows


def _run_key(**values: Any) -> str:
    import hashlib

    payload = json.dumps(values, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _aggregate(rows: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    if not rows:
        return {"question_count": 0, "metrics": {}}
    metric_type = (
        RetrievalQualityMetrics
        if mode == "retrieval_quality"
        else RetrievalMetrics
        if mode == "retrieval"
        else QuestionMetrics
    )
    names = tuple(metric_type.__dataclass_fields__)
    means: dict[str, float] = {}
    totals: dict[str, int] = {}
    for name in names:
        values = [row["metrics"][name] for row in rows]
        if name.endswith("_calls"):
            totals[name] = sum(int(value) for value in values)
        else:
            means[name] = sum(float(value) for value in values) / len(values)
    summary = {
        "question_count": len(rows),
        "metrics": {**means, **totals},
    }
    if mode not in {"retrieval", "retrieval_quality"}:
        summary["answer_and_evidence_quality_are_reported_separately"] = True
    return summary
