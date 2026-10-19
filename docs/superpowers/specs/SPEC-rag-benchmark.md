# Module Spec: `rag-benchmark`

## Objective
Tune and compare the standard vector and graph RAG pipelines with reranking enabled and disabled using `data/MultiHopRAG.json`, then compare the selected standard pipeline with its ReAct version using the same selected reranker setting.

## Interface
Load query, answer, question type, and evidence URLs. On development, run configured retrieval and reranking without answer generation; persist ranked evidence, retrieval scores, and latency. On validation and test, run the full retrieval/generation pipeline and also persist answers, citations, and answer scores. Record reranker state, configuration, and model IDs for every condition.

## Success Criteria
- Both backends use identical fixed data splits and comparable generation/reranking settings.
- Validation compares reranker on/off while holding other settings fixed; the final standard-versus-agentic run uses the same frozen reranker state.
- Report Evidence Recall@5/10, Precision@5/10, nDCG@10, latency, and call counts on development. Report answer Exact Match, token F1, and citation precision/recall in addition on validation and test.
- Retrieval tuning uses development data; full-pipeline comparison uses validation data; final standard-versus-agentic comparison uses the same untouched test questions.

## Boundaries
- Keep evidence and answer quality separate when evaluating complete pipelines on validation and test.
- Ask first before selecting a single weighted score when the measures conflict.
- Never tune against held-out test results or use evidence labels for indexing.

## Implementation choices
The benchmark uses stratified 70/15/15 development/validation/test partitions with seed 42 and writes a membership manifest, per-question JSONL records, condition summaries, and an aggregate summary beneath ignored `artifacts/benchmark/` output. Tune, compare, and final phases are restricted to development, validation, and test respectively. Retrieval-only tune records use a distinct mode and omit generated response and answer-quality fields.
