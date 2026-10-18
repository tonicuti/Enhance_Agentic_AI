# Design: RAG Retrieval Evaluation and ReAct Integration

## Objective

Build an evaluable retrieval system for questions over `data/corpus.json`. First compare and tune a Pinecone vector RAG pipeline and a Neo4j graph RAG pipeline. Then select the strongest standard RAG configuration and compare it with a ReAct-based Agentic RAG version. The evaluation dataset is `data/MultiHopRAG.json`.

Success is measured across answer quality and evidence retrieval. The final response contract is JSON with `role`, `context`, and `sources`, where `sources` is a list of article URL strings.

## Approved Architecture

The design uses shared corpus preparation and benchmark contracts with interchangeable retrieval backends. Both retrieval systems produce ranked text passages with document ID, title, URL, and retrieval score. A shared, optional Ollama reranker can rerank either backend's candidates. The generation model and evaluation questions remain fixed when comparing backends.

### Capability map and build sequence

The approved module boundaries and dependency order are recorded in [`CAPABILITY-MAP.md`](../../../CAPABILITY-MAP.md). The implementation sequence is ingestion, parallel vector and graph indexing/retrieval, optional reranking, benchmark and tuning, selected standard retriever, ReAct integration, response formatting, and a chat interface.

### Data flow

1. Load and validate article records from `data/corpus.json`.
2. Assign a stable document ID derived from the canonical article URL; preserve title, URL, source, category, date, and article text.
3. Split article text with a configurable splitter. Store each chunk with its document ID and source metadata.
4. Vector path: embed chunks with `embeddinggemma:latest`, index them in Pinecone, and retrieve candidates.
5. Graph path: extract entity and relationship records from corpus text with `qwen3:1.7b`, store article/chunk/entity nodes and provenance-linked relationships in Neo4j, and traverse the graph for query-relevant candidates.
6. During benchmark tuning, compare candidates with reranking enabled and disabled. If enabled, pass candidates from either backend through `dengcao/Qwen3-Reranker-0.6B:Q8_0`; if disabled, preserve the backend's candidate order. Freeze the selected setting for the standard RAG and ReAct retrieval pipelines.
7. For each benchmark question, pass ranked evidence to `qwen3:1.7b` for a grounded answer and record answer, cited URLs, retrieval ranks, reranker setting, configuration, and latency.
8. After selecting the strongest standard RAG configuration, expose it as a LangChain tool to a ReAct agent. The agent may reformulate the query and retrieve more evidence within a configurable step limit.

The graph is built from the corpus only. `evidence_list` in the evaluation data is used for scoring and must not be used to construct or tune the graph.

### Evaluation protocol

Use a deterministic, fixed split of `MultiHopRAG.json`, stratified by `question_type`. Tune on development data, select configurations on validation data, and reserve held-out test data for final reporting. Compare reranker enabled and disabled during tuning/validation. Apply the same selected splitter settings, reranker setting, generation model, and scoring rules to both vector and graph candidates where applicable; record the setting for every run. In the final benchmark, compare standard RAG with ReAct-based Agentic RAG using the same selected reranker setting.

Report:

- Evidence Recall@5 and Recall@10, plus nDCG@10.
- Normalized answer Exact Match and token-level F1.
- Citation precision and recall against gold evidence URLs.
- Latency and retrieval/model call counts.

Treat evidence and answer quality as co-primary measures. If a configuration does not lead on both, report the tradeoff instead of introducing an arbitrary weighted score. Once the retrieval configuration is frozen, run the final benchmark comparing the selected standard RAG system with its ReAct-based Agentic RAG version on the same held-out questions. Do not tune against held-out results.

### Response contract

```json
{
  "role": "assistant",
  "context": "Answer supported by retrieved evidence.",
  "sources": ["https://example.com/article"]
}
```

The `sources` list contains only URLs of retrieved articles used to support the answer. If evidence is insufficient, the system should state that in `context` and return an empty `sources` list.

### Reranker evaluation configuration

Reranker on/off is an evaluation setting. Benchmark tuning compares both modes while holding other settings fixed and records the setting in run metadata. After choosing a configuration, freeze that setting for the chat pipeline and use the same setting in standard RAG and ReAct retrieval. The chat interface does not expose a reranker toggle or accept a per-request override.

## Tech Stack

- LangChain for document processing, retriever/tool interfaces, and ReAct orchestration.
- Streamlit 1.59 or newer (below 2.0) for the chat interface.
- Ollama generation model: `qwen3:1.7b`.
- Ollama embedding model: `embeddinggemma:latest`.
- Ollama reranker model: `dengcao/Qwen3-Reranker-0.6B:Q8_0`.
- Pinecone for the vector-backed variant.
- Neo4j for the graph-backed variant.
- Python is the proposed implementation language, subject to confirmation before implementation.

## Proposed Commands

These are target CLI contracts; module/package names and flags should be finalized in the implementation plan:

```powershell
python -m rag_system.index --backend vector
python -m rag_system.index --backend graph
python -m rag_system.benchmark --phase tune --backend both --split development --reranker both
python -m rag_system.benchmark --phase compare --backend both --split validation --reranker both
python -m rag_system.benchmark --phase final --compare standard,agentic --backend selected --split test --reranker selected
python -m rag_system.chat --backend selected
```

## Proposed Project Structure

```text
data/                         Existing corpus and evaluation JSON files
configs/                      Model, splitter, retrieval, and evaluation settings
src/rag_system/               Application package
  ingestion/                   Validation, normalization, chunking
  retrievers/vector/            Pinecone indexing and retrieval
  retrievers/graph/             Neo4j extraction, indexing, and retrieval
  benchmark/                    Dataset loading, scoring, run records
  reranking/                    Optional candidate reranking and setting propagation
  agent/                        ReAct tools and orchestration
  response/                     JSON response validation
app.py                         Streamlit chat entry point
tests/unit/                     Offline unit tests
tests/integration/              Optional-service integration tests
tests/e2e/                      Small end-to-end pipeline checks
artifacts/                      Ignored local run results and generated indexes
```

## Code Style and Contract Example

Use type-annotated Python, explicit configuration objects, and keep retrieval backends behind the same result contract.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievedPassage:
    document_id: str
    title: str
    url: str
    text: str
    score: float
```

The result contract makes benchmark scoring and citation generation independent of the backend implementation.

## Reliability and Testing Strategy

- Validate required corpus fields (`title`, `url`, `body`) and report malformed record IDs.
- Validate extracted graph records and require source provenance for every stored fact/relationship.
- Use bounded timeouts and retries for Ollama, Pinecone, and Neo4j. Distinguish service errors from valid empty retrieval results.
- Unit checks: record parsing, deterministic IDs, chunk metadata, graph provenance validation, metric calculations, and JSON response validation.
- Reranker benchmark checks: enabled mode reranks, disabled mode preserves candidate order, and both standard/ReAct paths honor the frozen selected state.
- Integration checks: each backend's indexing and retrieval against a small corpus subset, with credentials and services supplied by the operator.
- End-to-end check: run a small fixed set through standard RAG, then through ReAct after integration.
- Full benchmark: run only after the split and configurations are frozen; persist per-question outputs and run metadata.

No tests or external services are run as part of this architecture draft.

## Boundaries

- **Always:** Preserve document URL/title metadata through chunking and retrieval; ground answers in retrieved text; keep evidence labels out of graph construction; record configuration, model identifiers, and reranker state for every benchmark run.
- **Ask first:** Change the held-out split after tuning begins; add a user interface or deployment target; add a new external service or paid dependency; use a different model for one backend; select a single winner when answer and evidence metrics conflict.
- **Never:** Use gold evidence to seed the retrieval index or graph; cite URLs that were not retrieved; silently count database/model failures as empty results; tune on held-out test results.

## Open Decisions Before Implementation

1. Python version, LangChain package versions, and exact CLI module names.
2. Pinecone and Neo4j deployment choices, index/graph schema details, credentials, and connection configuration.
3. Chunk splitter type, chunk size, overlap, candidate depth, reranking depth, and ReAct step limit.
4. Exact deterministic split percentages and random seed.
5. Graph extraction schema, entity resolution rules, and how relation confidence is used.
6. How to invoke and validate the selected Ollama reranker model through the chosen LangChain integration.
7. Deployment target and whether to add authentication.
8. Selection policy if vector and graph lead on different co-primary metrics.

## Success Criteria

- The same corpus can be indexed into both backends while preserving source metadata.
- The benchmark can run both retrievers on the same development and validation questions and emit comparable evidence and answer metrics.
- Each run is reproducible from recorded model names, configuration, split, and outputs.
- A selected standard RAG configuration can be compared against its ReAct-enhanced counterpart on the frozen test split.
- Final responses validate against the agreed JSON contract and cite only source URLs present in retrieved evidence.
