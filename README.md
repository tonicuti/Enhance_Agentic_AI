# Agentic AI for Enterprise

Hybrid RAG over `data/corpus.json`, with independently selectable Pinecone vector and Neo4j graph retrieval. The `data/MultiHopRAG.json` dataset is reserved for benchmark questions and scoring; its gold evidence is never used to build either index.

## Requirements

- Python 3.12 or newer
- Ollama with the configured generation, embedding, and (for reranker evaluations) reranking models
- Pinecone account and API key for Vector RAG
- Neo4j database and credentials for Graph RAG

Pull the local models:

```powershell
ollama pull qwen3:1.7b
ollama pull embeddinggemma:latest
ollama pull dengcao/Qwen3-Reranker-0.6B:Q8_0
```

Install the application using the existing `uv` project:

```powershell
uv sync
```

Or with pip:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

Copy `.env.example` to `.env` and fill in the Pinecone and Neo4j values you use. `Settings.from_env()` reads only this application's settings and does not export unrelated `.env` entries into the process.

## Build the retrieval indexes

Index each backend independently, or both in sequence:

```powershell
python -m agentic_ai_for_enterprise.index --backend vector
python -m agentic_ai_for_enterprise.index --backend graph
python -m agentic_ai_for_enterprise.index --backend both
```

Indexing reads only the corpus file. The defaults are `data/corpus.json`, `configs/default.json`, 1,000-character chunks, and 150-character overlap. Re-running indexing uses stable article/chunk IDs and updates the existing records.

## Benchmark retrieval and answers

The benchmark deterministically stratifies questions by `question_type` into development, validation, and held-out test partitions (70/15/15, seed 42). Development runs evaluate retrieval and reranking only; they do not generate answers. Validation runs compare complete RAG pipelines, including answer and citation quality. The test partition is reserved for the final comparison. Each phase uses its corresponding split.

Run a development sweep across vector and graph retrieval, with reranking off and on:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase tune --backend both --split development --reranker both
```

Compare candidates on validation:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase compare --backend both --split validation --reranker both
```

Use development retrieval metrics to narrow the candidates, then review evidence and answer metrics on validation. Set the chosen backend and frozen reranker state in `configs/default.json`. The final command compares standard RAG with Agentic RAG on the same held-out questions:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase final --backend selected --split test --reranker selected
```

For a short operational smoke run, add `--limit 10`. You can set `--run-id <name>` to resume a partially completed matching run. Do not use a limited run to select a winning configuration.

Each run writes a split manifest, per-question JSONL records, condition summaries, and an aggregate summary under `artifacts/benchmark/<run-id>/`. Development reports contain Evidence Recall@5/10, Precision@5/10, nDCG@10, latency, and retrieval/model-call counts. They have `mode: retrieval` and contain no generated response or answer-quality metrics. Validation and final reports also include normalized answer exact match, token F1, and citation precision/recall. Evidence and answer quality are reported separately; there is no weighted winner score.

## Chat

Start the Streamlit app after indexing the selected backend:

```powershell
python -m streamlit run app.py
```

The sidebar selects Vector RAG (Pinecone) or Graph RAG (Neo4j). Chat uses the Agentic RAG ReAct loop with the frozen backend and reranker values from `configs/default.json`. Reranker on/off is available only to benchmark runs. Responses contain `role`, `context`, and `sources`; citations link to retrieved article URLs.

## Configuration

`configs/default.json` controls the selected backend, frozen reranker state, chunking, candidate/context depths, graph traversal depth, ReAct step limit, and benchmark split. `.env` contains service connection details and model identifiers; use `.env.example` for the variable names and safe placeholders. Keep `.env` private.
