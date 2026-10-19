"""Behavior checks for development retrieval evaluation."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agentic_ai_for_enterprise.application import RagApplication
from agentic_ai_for_enterprise.benchmark.__main__ import _select_modes, main
from agentic_ai_for_enterprise.benchmark.runner import run_condition
from agentic_ai_for_enterprise.config import PipelineConfig
from agentic_ai_for_enterprise.contracts import (
    BenchmarkQuestion,
    PipelineResult,
    RagAnswer,
    RetrievedPassage,
    RetrievalTrace,
)


PASSAGE = RetrievedPassage(
    document_id="article-1",
    chunk_id="chunk-1",
    title="Example",
    url="https://example.com/evidence",
    text="Evidence text",
    score=0.8,
)
QUESTION = BenchmarkQuestion(
    question_id="question-1",
    query="What happened?",
    answer="An answer",
    question_type="comparison",
    evidence_urls=(PASSAGE.url,),
)


class RetrievalTuningTests(unittest.TestCase):
    def test_tune_selects_retrieval_mode(self) -> None:
        self.assertEqual(_select_modes("tune", "standard,agentic"), ["retrieval"])
        self.assertEqual(_select_modes("compare", "standard,agentic"), ["standard"])

    def test_development_split_is_reserved_for_tuning(self) -> None:
        for phase, split in (("tune", "validation"), ("compare", "development")):
            with self.subTest(phase=phase), patch.object(
                sys, "argv", ["benchmark", "--phase", phase, "--split", split]
            ), patch(
                "agentic_ai_for_enterprise.benchmark.__main__.Settings.from_env",
                return_value=SimpleNamespace(),
            ):
                with self.assertRaises(SystemExit):
                    main()

    def test_application_retrieve_does_not_generate_answer(self) -> None:
        class FakeRetriever:
            model_call_count = 0

            def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]:
                self.model_call_count += 2
                return [PASSAGE]

            def close(self) -> None:
                pass

        retriever = FakeRetriever()
        with patch("agentic_ai_for_enterprise.application.get_retriever", return_value=retriever), patch(
            "agentic_ai_for_enterprise.application.generate_answer",
            side_effect=AssertionError("generation must not run"),
        ):
            application = RagApplication(
                backend="vector", settings=SimpleNamespace(), pipeline=PipelineConfig()
            )
            trace = application.retrieve(QUESTION.query)

        self.assertEqual(trace.passages, (PASSAGE,))
        self.assertEqual(trace.retrieval_calls, 1)
        self.assertEqual(trace.model_calls, 2)

    def test_tune_writes_only_retrieval_metrics_and_resumes(self) -> None:
        calls = []

        class FakeApplication:
            def __init__(self, **kwargs: object) -> None:
                pass

            def retrieve(self, query: str) -> RetrievalTrace:
                calls.append(query)
                return RetrievalTrace(
                    passages=(PASSAGE,), retrieval_calls=1, model_calls=0, elapsed_seconds=0.1
                )

            def answer(self, query: str, *, mode: str) -> None:
                raise AssertionError("generation must not run during tuning")

            def close(self) -> None:
                pass

        settings = SimpleNamespace(embedding_model="embed", reranker_model="rerank")
        with tempfile.TemporaryDirectory() as directory, patch(
            "agentic_ai_for_enterprise.benchmark.runner.RagApplication", FakeApplication
        ):
            kwargs = dict(
                questions=[QUESTION],
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                phase="tune",
                split_name="development",
                backend="vector",
                mode="retrieval",
                pipeline=PipelineConfig(),
                settings=settings,
                output_dir=Path(directory),
                run_id="retrieval-test",
            )
            summary = run_condition(**kwargs)
            resumed = run_condition(**kwargs)
            row = json.loads(Path(summary["question_results"]).read_text(encoding="utf-8"))

        self.assertEqual(calls, [QUESTION.query])
        self.assertEqual(summary["question_count"], 1)
        self.assertEqual(resumed["question_count"], 1)
        self.assertNotIn("answer_exact_match", summary["metrics"])
        self.assertNotIn("answer_and_evidence_quality_are_reported_separately", summary)
        self.assertEqual(row["metrics"]["evidence_recall_at_5"], 1.0)
        self.assertEqual(row["metrics"]["ndcg_at_10"], 1.0)
        self.assertNotIn("answer_exact_match", row["metrics"])
        self.assertNotIn("response", row)
        self.assertNotIn("gold_answer", row)
        self.assertNotIn("generation", row["models"])

    def test_validation_keeps_answer_and_citation_metrics(self) -> None:
        class FakeApplication:
            def __init__(self, **kwargs: object) -> None:
                pass

            def answer(self, query: str, *, mode: str) -> PipelineResult:
                return PipelineResult(
                    answer=RagAnswer(role="assistant", context=QUESTION.answer, sources=[PASSAGE.url]),
                    passages=(PASSAGE,),
                    retrieval_calls=1,
                    model_calls=1,
                    elapsed_seconds=0.2,
                )

            def close(self) -> None:
                pass

        settings = SimpleNamespace(
            generation_model="generate", embedding_model="embed", reranker_model="rerank"
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "agentic_ai_for_enterprise.benchmark.runner.RagApplication", FakeApplication
        ):
            summary = run_condition(
                questions=[QUESTION],
                dataset_fingerprint="dataset",
                split_fingerprint="split",
                phase="compare",
                split_name="validation",
                backend="vector",
                mode="standard",
                pipeline=PipelineConfig(),
                settings=settings,
                output_dir=Path(directory),
                run_id="validation-test",
            )
            row = json.loads(Path(summary["question_results"]).read_text(encoding="utf-8"))

        self.assertEqual(row["response"]["context"], QUESTION.answer)
        self.assertEqual(row["metrics"]["answer_exact_match"], 1.0)
        self.assertEqual(row["metrics"]["citation_recall"], 1.0)
        self.assertEqual(row["models"]["generation"], "generate")


if __name__ == "__main__":
    unittest.main()
