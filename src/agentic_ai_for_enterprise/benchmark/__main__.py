"""CLI entry point for tuning and comparing retrieval pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import replace
from pathlib import Path

from ..config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, PipelineConfig, Settings
from .dataset import load_questions
from .runner import run_condition
from .splits import split_questions


LOGGER = logging.getLogger("rag_system.benchmark")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tune", "compare", "final", "hybrid"), required=True)
    parser.add_argument("--backend", choices=("vector", "graph", "hybrid", "both", "selected"))
    parser.add_argument("--split", choices=("development", "validation", "test"))
    parser.add_argument("--reranker", choices=("enabled", "disabled", "both", "selected"), default="both")
    parser.add_argument("--compare", help="Generation modes for compare/final phases (standard,agentic).")
    parser.add_argument("--limit", type=int, help="Run only the first N questions from the deterministic split.")
    parser.add_argument("--run-id", help="Reuse this ID to resume matching JSONL results.")
    parser.add_argument("--dataset", type=Path, default=PROJECT_ROOT / "data" / "MultiHopRAG.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "artifacts" / "benchmark")
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    pipeline = PipelineConfig.load(args.config)
    settings = Settings.from_env()
    backend_choice = args.backend or (
        "selected" if args.phase == "final" else "hybrid" if args.phase == "hybrid" else "both"
    )
    expected_split = {
        "tune": "development",
        "compare": "validation",
        "final": "test",
        "hybrid": "development",
    }[args.phase]
    split_name = args.split or expected_split
    if split_name != expected_split:
        raise SystemExit(f"The {args.phase} phase must use the {expected_split} split.")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be at least 1.")
    questions, dataset_fingerprint = load_questions(args.dataset)
    splits = split_questions(questions, pipeline)
    selected_questions = list(getattr(splits, split_name))
    if args.limit is not None:
        selected_questions = selected_questions[: args.limit]
    if not selected_questions:
        raise SystemExit(f"The deterministic {split_name} split contains no questions.")

    if args.phase == "final" and backend_choice == "both":
        raise SystemExit("The final phase uses one selected backend; choose --backend selected.")
    if args.phase == "hybrid" and backend_choice != "hybrid":
        raise SystemExit("The hybrid phase requires --backend hybrid.")
    if args.phase == "hybrid" and args.reranker == "selected":
        raise SystemExit("The hybrid phase requires --reranker enabled, disabled, or both.")
    backends = _select_backends(backend_choice, pipeline)
    reranker_modes = _select_rerankers(args.reranker, pipeline, args.phase)
    modes = _select_modes(args.phase, args.compare)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    run_directory = args.output_dir / run_id
    run_directory.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "dataset_fingerprint": dataset_fingerprint,
        "split_fingerprint": splits.fingerprint,
        "seed": splits.seed,
        "fractions": {
            "development": pipeline.development_fraction,
            "validation": pipeline.validation_fraction,
            "test": pipeline.test_fraction,
        },
        "counts": {
            "development": len(splits.development),
            "validation": len(splits.validation),
            "test": len(splits.test),
        },
        "membership": splits.membership,
    }
    (run_directory / "split-manifest.json").write_text(
        json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summaries = []
    for backend in backends:
        for reranker_enabled in reranker_modes:
            condition_config = replace(pipeline, selected_backend=backend, reranker_enabled=reranker_enabled)
            condition_config.validate()
            for mode in modes:
                summaries.append(
                    run_condition(
                        questions=selected_questions,
                        dataset_fingerprint=dataset_fingerprint,
                        split_fingerprint=splits.fingerprint,
                        phase=args.phase,
                        split_name=split_name,
                        backend=backend,
                        mode=mode,
                        pipeline=condition_config,
                        settings=settings,
                        output_dir=run_directory,
                        run_id=run_id,
                    )
                )
    summary_path = run_directory / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    LOGGER.info("Saved %d benchmark condition(s) to %s", len(summaries), summary_path)
    return 0


def _select_backends(choice: str, pipeline: PipelineConfig) -> list[str]:
    if choice == "both":
        return ["vector", "graph"]
    return [pipeline.selected_backend if choice == "selected" else choice]


def _select_rerankers(choice: str, pipeline: PipelineConfig, phase: str) -> list[bool]:
    if choice == "both":
        if phase == "final":
            raise SystemExit("The final phase must use --reranker selected to preserve the frozen setting.")
        return [False, True]
    if choice == "selected":
        return [pipeline.reranker_enabled]
    return [choice == "enabled"]


def _select_modes(phase: str, compare: str | None) -> list[str]:
    if phase == "tune":
        return ["retrieval"]
    if phase == "hybrid":
        if compare is not None:
            raise SystemExit("The hybrid phase is retrieval-only and does not accept --compare.")
        return ["retrieval_quality"]
    modes = [item.strip() for item in (compare or "standard,agentic").split(",") if item.strip()]
    if not modes or any(mode not in {"standard", "agentic"} for mode in modes):
        raise SystemExit("--compare must be a comma-separated selection of standard,agentic.")
    if phase == "final" and set(modes) != {"standard", "agentic"}:
        raise SystemExit("The final phase compares standard and agentic RAG together.")
    return list(dict.fromkeys(modes))


if __name__ == "__main__":
    raise SystemExit(main())
