# Agentic RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. This session uses native, single-agent execution.

**Goal:** Implement the approved internal-document RAG system with separately evaluable Pinecone vector and Neo4j graph retrieval, benchmark-only reranker comparisons, a bounded LangChain ReAct retrieval loop, and a Streamlit chat interface.

**Architecture:** Shared typed document, passage, answer, and configuration contracts feed interchangeable vector and graph retrievers. Offline ingestion and evaluation logic stay separate from service adapters; benchmark settings are explicit and persisted, while chat loads a frozen selected configuration and never exposes reranker controls.

**Tech Stack:** Python 3.12+, LangChain, `langchain-ollama`, `langchain-pinecone`, Ollama (`qwen3:1.7b`, `embeddinggemma:latest`, `dengcao/Qwen3-Reranker-0.6B:Q8_0`), Pinecone, Neo4j Python Driver, Streamlit, and the existing `uv` project scaffold under `src/agentic_ai_for_enterprise`.

**Spec:** `docs/superpowers/specs/2026-10-18-rag-retrieval-architecture-design.md` and the module specs in `docs/superpowers/specs/`.

## Global Constraints

- Read only article records from `data/corpus.json` for indexing and graph construction.
- Use `data/MultiHopRAG.json` only for benchmark queries and scoring; never ingest `evidence_list` into an index or graph.
- Preserve document ID, title, URL, source, category, date, and chunk index across ingestion and retrieval.
- Use `qwen3:1.7b` for generation and graph extraction; use `embeddinggemma:latest` for embeddings; use `dengcao/Qwen3-Reranker-0.6B:Q8_0` for evaluation reranking.
- Compare reranking enabled and disabled during benchmark tuning/validation; freeze the selected state for standard and ReAct chat pipelines.
- Use Pinecone for vector storage and Neo4j for graph storage; keep secrets in environment variables and never in source or artifacts.
- Final response JSON must contain `role`, `context`, and `sources`; sources are retrieved article URL strings only.
- Use a deterministic stratified 70/15/15 development/validation/test split with seed 42, recorded with each run.
- Do not add or run automated tests in this implementation unless requested; use syntax/CLI checks and manual inspection as permitted verification.

## Review Focus

- Malformed corpus records: report record indices and reasons; valid records remain indexable.
- Duplicate/canonical URLs: stable IDs and re-indexing must be idempotent.
- Empty, malformed, or prompt-injection-like model extraction output: validate schema and retain provenance; never execute model output as a query.
- Missing service credentials or unavailable Ollama/Pinecone/Neo4j: fail with an actionable service-specific message, not an empty successful retrieval.
- Empty evidence or malformed model answer: return an insufficient-evidence response with `sources: []` and never fabricate citations.

## File Structure

- `src/agentic_ai_for_enterprise/config.py`: environment-backed settings and benchmark defaults.
- `src/agentic_ai_for_enterprise/contracts.py`: dataclasses and protocols shared by all modules.
- `src/agentic_ai_for_enterprise/ingestion/`: corpus parsing, canonical IDs, and LangChain text splitting.
- `src/agentic_ai_for_enterprise/retrievers/vector/`: Pinecone lifecycle, indexing, and retrieval.
- `src/agentic_ai_for_enterprise/retrievers/graph/`: extraction schema, Neo4j persistence, and graph candidate retrieval.
- `src/agentic_ai_for_enterprise/reranking/`: shared optional Ollama reranker.
- `src/agentic_ai_for_enterprise/response/`: grounded answer generation and response validation.
- `src/agentic_ai_for_enterprise/agent/`: bounded ReAct prompt/tool loop over the selected retriever.
- `src/agentic_ai_for_enterprise/benchmark/`: dataset parsing, fixed splits, metrics, and JSONL run artifacts.
- `src/agentic_ai_for_enterprise/index.py`, `benchmark/__main__.py`: documented CLI entry points.
- `app.py`: Streamlit chat, selected backend, and response/source rendering only.
- `configs/default.json`: selected chat backend and frozen reranker/retrieval/model settings.
- `.env.example`, `requirements.txt`, `README.md`: setup and operation instructions.

## Task List

### Task 1: Runtime configuration and shared contracts

**Files:** Modify `pyproject.toml`, `requirements.txt`, and `src/agentic_ai_for_enterprise/__init__.py`; create `src/agentic_ai_for_enterprise/config.py`, `src/agentic_ai_for_enterprise/contracts.py`, `configs/default.json`, `.env.example`.

**Interfaces:** `Settings.from_env() -> Settings`; `Article`, `TextChunk`, `RetrievedPassage`, `RagAnswer`, `Retriever` protocol; `RerankerConfig`, `RetrievalConfig`, and `BenchmarkConfig` are explicit immutable configuration values.

- [x] Define environment variable names for Ollama base URL, Pinecone API key/index/namespace, Neo4j URI/user/password/database, and selected pipeline settings.
- [x] Add bounded dependency ranges for supported LangChain integration packages, Pinecone, Neo4j, Streamlit, and dotenv/config parsing.
- [x] Define config defaults: chunk size 1000, overlap 150, candidate depth 20, final context depth 5, graph traversal depth 2, ReAct retrieval limit 4, and model IDs listed in the approved spec.
- [x] Ensure config loading rejects invalid ranges and missing required values only when the corresponding external backend is selected.
- [x] Verify modules compile and config serialization contains no secrets.

### Task 2: Corpus ingestion and vector retrieval

**Files:** Create `src/agentic_ai_for_enterprise/ingestion/__init__.py`, `src/agentic_ai_for_enterprise/ingestion/corpus.py`, `src/agentic_ai_for_enterprise/ingestion/chunking.py`, `src/agentic_ai_for_enterprise/retrievers/__init__.py`, `src/agentic_ai_for_enterprise/retrievers/vector/__init__.py`; create `src/agentic_ai_for_enterprise/retrievers/vector/pinecone_store.py`, `src/agentic_ai_for_enterprise/index.py`.

**Interfaces:** `load_corpus(path) -> tuple[list[Article], list[ValidationIssue]]`; `chunk_articles(articles, config) -> list[TextChunk]`; vector adapter `index(chunks) -> IndexSummary`, `retrieve(query, limit) -> list[RetrievedPassage]`.

- [x] Parse JSON as an array and validate required `title`, `url`, and `body`, retaining optional source/category/publication metadata.
- [x] Canonicalize URLs for deterministic SHA-256 document IDs and assign deterministic chunk IDs from document ID plus chunk index.
- [x] Split with LangChain `RecursiveCharacterTextSplitter`; copy article metadata onto every chunk.
- [x] Connect with `OllamaEmbeddings(model="embeddinggemma:latest")` and `PineconeVectorStore`; create or validate configured index dimension/metric without exposing keys.
- [x] Upsert with stable chunk IDs and return ranked passages with similarity scores and source metadata.
- [x] Implement `python -m agentic_ai_for_enterprise.index --backend vector`; malformed articles are summarized and never silently dropped.
- [x] Verify CLI help and Python compilation; do not contact external services as part of syntax checks.

### Task 3: Graph extraction, Neo4j indexing, and retrieval

**Files:** Create `src/agentic_ai_for_enterprise/retrievers/graph/__init__.py`, `src/agentic_ai_for_enterprise/retrievers/graph/schema.py`, `src/agentic_ai_for_enterprise/retrievers/graph/extraction.py`, `src/agentic_ai_for_enterprise/retrievers/graph/neo4j_store.py`.

**Interfaces:** `extract_graph_records(chunks, llm) -> list[GraphRecord]`; `GraphStore.index(chunks, records) -> IndexSummary`; `GraphStore.retrieve(query, limit, traversal_depth) -> list[RetrievedPassage]`.

- [x] Define typed entity/relation records that require chunk ID provenance and a bounded confidence value.
- [x] Extract entities and relations with `ChatOllama(model="qwen3:1.7b")`, validate structured output, and skip invalid records with clear diagnostics.
- [x] Persist `Article`, `Chunk`, and `Entity` nodes and provenance-linked relations with parameterized Cypher and deterministic merge keys.
- [x] Use Neo4j full-text matching for query entities and bounded graph traversal; do not add vector search to the graph variant.
- [x] Rank graph candidates deterministically using entity/relation match count and confidence; retain URL and passage metadata.
- [x] Make graph construction consume only corpus chunks, never evaluation evidence labels.
- [x] Extend index CLI for `--backend graph` and `--backend both`; verify help and syntax without connecting to Neo4j or Ollama.

### Task 4: Shared benchmark-only reranking and grounded response generation

**Files:** Create `src/agentic_ai_for_enterprise/reranking/__init__.py`, `src/agentic_ai_for_enterprise/reranking/ollama.py`, `src/agentic_ai_for_enterprise/response/__init__.py`, `src/agentic_ai_for_enterprise/response/answer.py`, `src/agentic_ai_for_enterprise/response/validation.py`.

**Interfaces:** `rerank(query, candidates, config) -> list[RetrievedPassage]`; `generate_answer(question, evidence, llm) -> RagAnswer`; `validate_answer(answer, evidence_urls) -> RagAnswer`.

- [x] Implement explicit benchmark reranker enabled/disabled configuration; disabled mode preserves original order and never loads/calls the model.
- [x] Use the configured Ollama reranker through a LangChain chat interface to score query-passage relevance; parse bounded scores and use stable tie-breaking.
- [x] Preserve provenance and retrieval metadata when reordering candidates.
- [x] Generate answers only from retrieved evidence with `qwen3:1.7b`; instruct it to abstain if evidence is insufficient.
- [x] Validate response fields and intersect proposed sources with URLs actually retrieved; invalid/unsupported outputs become insufficient-evidence answers with empty sources.
- [x] Verify compilation and inspect the fixed response contract.

### Task 5: Reproducible benchmark and metrics

**Files:** Create `src/agentic_ai_for_enterprise/benchmark/__init__.py`, `src/agentic_ai_for_enterprise/benchmark/dataset.py`, `src/agentic_ai_for_enterprise/benchmark/splits.py`, `src/agentic_ai_for_enterprise/benchmark/metrics.py`, `src/agentic_ai_for_enterprise/benchmark/runner.py`, `src/agentic_ai_for_enterprise/benchmark/__main__.py`.

**Interfaces:** `load_questions(path) -> list[BenchmarkQuestion]`; `split_questions(questions, seed=42) -> DatasetSplits`; `score_run(question, passages, answer) -> QuestionMetrics`; `run_benchmark(config) -> BenchmarkSummary`.

- [x] Validate `query`, `answer`, `question_type`, and evidence URLs; retain gold evidence only in benchmark objects.
- [x] Deterministically stratify 70/15/15 by question type, preserving all examples and recording seed, split membership, and dataset fingerprint.
- [x] Implement Evidence Recall@5/10, nDCG@10, normalized answer Exact Match, token F1, citation precision/recall, latency, retrieval calls, and model calls.
- [x] Support vector/graph, development/validation/test, and reranker `enabled|disabled|both|selected` CLI options; `both` runs paired configs with other parameters held fixed.
- [x] Prevent tuning on test split; final standard-versus-agentic comparison reads frozen selected config and the same held-out questions.
- [x] Persist per-question JSONL and aggregate JSON under ignored `artifacts/`, including all models/configuration and reranker state.
- [x] Verify CLI help, deterministic split configuration by manual inspection, and Python compilation.

### Task 6: Bounded ReAct retrieval orchestration

**Files:** Create `src/agentic_ai_for_enterprise/agent/__init__.py`, `src/agentic_ai_for_enterprise/agent/react.py`, `src/agentic_ai_for_enterprise/agent/prompts.py`.

**Interfaces:** `answer_with_react(question, retriever, llm, config) -> AgentRun`; `AgentRun` contains the validated answer, tool-call count, and retrieval trace.

- [x] Define a LangChain retrieval tool that accepts only a bounded query string and returns compact passage records.
- [x] Use an explicit ReAct prompt with Thought/Action/Observation/Final Answer instructions and evidence-only citation rules.
- [x] Bound retrieval iterations by config; parse tool actions defensively, reject unknown actions, and stop safely on malformed model output.
- [x] Reuse the frozen selected reranker configuration for every retrieval call; do not let the model or UI modify it.
- [x] Accumulate provenance from observed passages and validate final citations against that set.
- [x] Verify syntax and inspect that ReAct termination is bounded.

### Task 7: Connect the Streamlit chat to retrieval

**Files:** Modify `app.py`; create `src/agentic_ai_for_enterprise/application.py`.

**Interfaces:** `answer_question(question, backend, mode="agentic") -> RagAnswer` wires settings, selected backend, reranker config, and ReAct; `application.get_retriever(backend)` creates the requested backend adapter.

- [x] Remove setup placeholder and call the retrieval/application service for submitted questions.
- [x] Keep backend selection and conversation clear action; do not add a reranker toggle, reranker status, or request-level reranker setting.
- [x] Render the agreed `role`, `context`, and URL-list `sources` contract with safe text rendering.
- [x] Display service errors as concise actionable messages without exposing secrets or stack traces; preserve chat session history.
- [x] Keep lazy imports/connections so app startup remains clear when a selected backend is not configured.
- [x] Verify Python compilation and perform a manual UI code-path inspection; service calls require operator-provided local Ollama and database credentials.

### Task 8: Setup and operator documentation

**Files:** Modify `README.md`, `requirements.txt`, `.gitignore`; create or update `CAPABILITY-MAP.md` if interfaces changed.

- [x] Document Python/Ollama model setup, Pinecone and Neo4j environment variables, indexing commands, benchmark phases, and Streamlit launch.
- [x] Explain that reranker comparisons occur only in evaluation and the selected state is frozen for chat and ReAct runs.
- [x] Document benchmark defaults, metric outputs, held-out test protections, and expected artifact files.
- [x] Ensure `.env`, secrets, and generated benchmark artifacts remain ignored; include only placeholder values in `.env.example`.
- [x] Verify documentation commands match the implemented CLI help output.

## Checkpoints

### Checkpoint A: Data and backends (Tasks 1-3)

- [x] Shared contracts are used by both backend adapters.
- [x] Vector and graph indexing/retrieval have separate CLI paths and the common passage contract.
- [x] No gold evidence data flows into indexing or graph construction.

### Checkpoint B: Answering and benchmark (Tasks 4-5)

- [x] Benchmark supports paired reranker comparisons and persists reproducible run metadata.
- [x] Standard answer citations can only refer to retrieved URLs.
- [x] Test split cannot be selected for tuning.

### Checkpoint C: Agent and UI (Tasks 6-8)

- [x] ReAct loop is bounded and uses the same frozen retriever/reranker configuration.
- [x] Chat uses the selected backend and contains no reranker setting control.
- [x] Setup and benchmark instructions match the code interfaces.

## Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| `qwen3:1.7b` may not reliably emit graph JSON or ReAct actions | Invalid graph facts or premature loop termination | Strict schemas, bounded retries, explicit parser, and diagnostics; do not treat malformed output as valid data |
| 2,556 questions across remote/local model and database calls can make evaluation expensive | Slow or interrupted benchmark | Support split/limit selection, resumable per-question JSONL, paired configs, and persisted run IDs |
| Embedding dimensions can differ from pre-existing Pinecone index | Indexing/search failure | Validate index dimension against an embedding probe and fail with a clear migration instruction |
| The dataset's gold evidence may not be fully present in the corpus | Metric ceiling below 1.0 | Report benchmark results as retrieval/answer measurements; never insert evidence labels into stores |
| External services and credentials are not available in the workspace | End-to-end run cannot complete here | Implement connection validation and document service prerequisites; do not claim a live benchmark ran |

## Open Questions

- Selection when evidence and answer metrics disagree remains governed by the approved spec: report the tradeoff and do not invent a weighted winner.
- Pinecone cloud/index deployment and Neo4j instance/credentials are operator configuration; defaults belong in `.env.example`, not source.
- The implementation defaults for splitter, retrieval depth, ReAct limit, and dataset split are recorded above and can be tuned by the benchmark.

## Follow-up: Hybrid BM25 + Vector Retrieval

The approved follow-up design and implementation plan are `docs/superpowers/specs/2026-10-22-hybrid-bm25-vector-design.md` and `docs/superpowers/plans/2026-10-22-hybrid-bm25-vector.md`.

- [x] Add local SQLite FTS5 BM25 indexing alongside Pinecone vector indexing.
- [x] Fuse vector and BM25 candidates with RRF before applying the configured reranker.
- [x] Make the hybrid backend selectable in standard and ReAct chat retrieval.
- [x] Add a development-only hybrid retrieval benchmark reporting Evidence Recall@5/10, Precision@5/10, and nDCG@10 only.
- [x] Update operator documentation and the capability map.
