# Graph Report - Agentic_AI_for_Enterprise  (2026-11-11)

## Corpus Check
- cluster-only mode — file stats not available

## Summary
- 302 nodes · 836 edges · 12 communities (9 shown, 3 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 125 edges (avg confidence: 0.93)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `af30aee0`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- neo4j_store.py
- Settings
- contracts.py
- PipelineConfig
- runner.py
- react.py
- config.py
- application.py
- SQLiteBM25Store
- agentic-ai-for-enterprise

## God Nodes (most connected - your core abstractions)
1. `Settings` - 40 edges
2. `RetrievedPassage` - 37 edges
3. `PipelineConfig` - 36 edges
4. `run_condition()` - 24 edges
5. `Neo4jGraphStore` - 21 edges
6. `TextChunk` - 18 edges
7. `main()` - 18 edges
8. `RagApplication` - 17 edges
9. `SQLiteBM25Store` - 16 edges
10. `answer_with_react()` - 16 edges

## Surprising Connections (you probably didn't know these)
- `_application_for_backend()` --uses--> `Settings`  [INFERRED]
  app.py → src/agentic_ai_for_enterprise/config.py
- `RetrievalTuningTests` --uses--> `RetrievedPassage`  [INFERRED]
  tests/test_retrieval_tuning.py → src/agentic_ai_for_enterprise/contracts.py
- `RetrievalTuningTests` --uses--> `PipelineConfig`  [INFERRED]
  tests/test_retrieval_tuning.py → src/agentic_ai_for_enterprise/config.py
- `RetrievalTuningTests` --uses--> `RagAnswer`  [INFERRED]
  tests/test_retrieval_tuning.py → src/agentic_ai_for_enterprise/contracts.py
- `_application_for_backend()` --uses--> `RagApplication`  [INFERRED]
  app.py → src/agentic_ai_for_enterprise/application.py

## Import Cycles
- None detected.

## Communities (12 total, 3 thin omitted)

### Community 0 - "neo4j_store.py"
Cohesion: 0.11
Nodes (17): TextChunk, build_parser(), main(), extract_graph_records(), GraphExtractionError, GraphExtractor, _chunk_row(), _escape_fulltext_term() (+9 more)

### Community 1 - "Settings"
Cohesion: 0.08
Nodes (10): ConfiguredRetriever, get_retriever(), _GraphRetriever, Settings, RetrievedPassage, Retriever, OllamaReranker, rerank() (+2 more)

### Community 2 - "contracts.py"
Cohesion: 0.10
Nodes (9): load_questions(), Article, chunk_articles(), canonicalize_url(), document_id_for_url(), load_corpus(), _optional_string(), ValidationIssue (+1 more)

### Community 3 - "PipelineConfig"
Cohesion: 0.10
Nodes (12): _application_for_backend(), build_parser(), main(), _select_backends(), _select_modes(), _select_rerankers(), _allocate_counts(), DatasetSplits (+4 more)

### Community 4 - "runner.py"
Cohesion: 0.15
Nodes (19): _ndcg(), normalize_answer(), _precision(), QuestionMetrics, _recall(), RetrievalMetrics, RetrievalQualityMetrics, score_retrieval() (+11 more)

### Community 5 - "react.py"
Cohesion: 0.12
Nodes (14): AgentRun, _answer_from_final(), answer_with_react(), _content_text(), FinalAgentAnswer, _parse_action(), _parse_final_answer(), RetrievalToolInput (+6 more)

### Community 6 - "config.py"
Cohesion: 0.10
Nodes (3): _chunk_metadata(), _metadata_string(), PineconeRetriever

### Community 7 - "application.py"
Cohesion: 0.13
Nodes (7): answer_question(), RagApplication, PipelineResult, RetrievalTrace, RetrievalTuningTests, retrieve(), answer()

### Community 8 - "SQLiteBM25Store"
Cohesion: 0.18
Nodes (3): _match_query(), _optional_string(), SQLiteBM25Store

## Knowledge Gaps
- **1 isolated node(s):** `agentic-ai-for-enterprise`
  These have ≤1 connection - possible missing edges. (Counts symbols only; 101 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **3 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Settings` connect `Settings` to `neo4j_store.py`, `PipelineConfig`, `runner.py`, `react.py`, `config.py`, `application.py`?**
  _High betweenness centrality (0.141) - this node is a cross-community bridge._
- **Are the 18 inferred relationships involving `Settings` (e.g. with `_application_for_backend()` and `answer_with_react()`) actually correct?**
  _`Settings` has 18 INFERRED edges - model-reasoned connections that need verification._
- **What connects `agentic-ai-for-enterprise` to the rest of the system?**
  _1 weakly-connected nodes found - possible missing edges._
- **Should `neo4j_store.py` be split into smaller, more focused modules?**
  _Cohesion score 0.10730804810360776 - nodes in this community are weakly interconnected._
- **Why does `RetrievedPassage` connect `Settings` to `neo4j_store.py`, `contracts.py`, `runner.py`, `react.py`, `config.py`, `application.py`, `SQLiteBM25Store`?**
  _High betweenness centrality (0.140) - this node is a cross-community bridge._
- **Are the 15 inferred relationships involving `RetrievedPassage` (e.g. with `AgentRun` and `_answer_from_final()`) actually correct?**
  _`RetrievedPassage` has 15 INFERRED edges - model-reasoned connections that need verification._
- **Should `Settings` be split into smaller, more focused modules?**
  _Cohesion score 0.08416389811738649 - nodes in this community are weakly interconnected._