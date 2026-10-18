# Module Spec: `rag-benchmark`

## Objective
Tune and compare the standard vector and graph RAG pipelines with reranking enabled and disabled using `data/MultiHopRAG.json`, then compare the selected standard pipeline with its ReAct version using the same selected reranker setting.

## Interface
Load query, answer, question type, and evidence URLs; call a configured retrieval/generation pipeline; persist ranked evidence, answer, citations, reranker enabled/disabled state, configuration, model IDs, scores, and latency.

## Success Criteria
- Both backends use identical fixed data splits and comparable generation/reranking settings.
- Validation compares reranker on/off while holding other settings fixed; the final standard-versus-agentic run uses the same frozen reranker state.
- Report Evidence Recall@5/10, nDCG@10, normalized answer Exact Match, token F1, citation precision/recall, latency, and call counts.
- Tuning uses development/validation data; final standard-versus-agentic comparison uses the same untouched test questions.

## Boundaries
- Always keep evidence and answer quality as co-primary measures.
- Ask first before selecting a single weighted score when the measures conflict.
- Never tune against held-out test results or use evidence labels for indexing.

## Implementation choices
The benchmark uses stratified 70/15/15 development/validation/test partitions with seed 42 and writes a membership manifest, per-question JSONL records, condition summaries, and an aggregate summary beneath ignored `artifacts/benchmark/` output. Test questions are rejected for tune and compare phases.
