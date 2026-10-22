# Agentic AI for Enterprise

Agentic RAG chat over `data/corpus.json`, with selectable Pinecone vector, Neo4j graph, and hybrid BM25 + vector retrieval. Hybrid search fuses lexical and semantic candidates before applying the configured reranker. The `data/MultiHopRAG.json` dataset is reserved for benchmark questions and scoring; its gold evidence is never used to build an index.

## Requirements

- Python 3.12 or newer
- A Python SQLite build with FTS5 support for Hybrid RAG
- Ollama with the configured generation and embedding models
- `llama-server` from llama.cpp for BGE reranking
- Pinecone account and API key for Vector RAG
- Neo4j database and credentials for Graph RAG

Pull the local models:

```powershell
ollama pull qwen3:1.7b
ollama pull embeddinggemma:latest
ollama pull qllama/bge-reranker-v2-m3:latest
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

## Start the BGE reranker

Ollama does not expose a pairwise reranking API for this BGE model. Run its Ollama-stored GGUF file through llama.cpp's reranking endpoint instead. Find the model file path and start `llama-server` in PowerShell:

```powershell
$modelLine = (ollama show --modelfile qllama/bge-reranker-v2-m3:latest | Select-String '^FROM ').Line
$modelPath = $modelLine -replace '^FROM\s+', ''
llama-server.exe -m $modelPath --embedding --pooling rank --reranking --host 127.0.0.1 --port 8080
```

Keep that server running while the application uses reranking. The app sends pairwise query/document batches to `http://localhost:8080/v1/rerank`, configured by `RERANKER_BASE_URL`. Ollama continues to serve generation and embeddings. Install a llama.cpp build that includes `llama-server.exe` before starting this service.

## Build the retrieval indexes

Index each backend independently. Hybrid indexing writes the corpus chunks to both Pinecone and a local SQLite FTS5 BM25 index; `both` continues to index vector and graph stores:

```powershell
python -m agentic_ai_for_enterprise.index --backend vector
python -m agentic_ai_for_enterprise.index --backend graph
python -m agentic_ai_for_enterprise.index --backend hybrid
python -m agentic_ai_for_enterprise.index --backend both
```

Indexing reads only the corpus file. The defaults are `data/corpus.json`, `configs/default.json`, 1,000-character chunks, and 150-character overlap. Re-running indexing uses stable article/chunk IDs and replaces the local BM25 contents with the current corpus chunks. Its default database path is `artifacts/index/hybrid_bm25.sqlite3` and can be changed with `hybrid_bm25_index_path` in `configs/default.json`.

## Benchmark retrieval and answers

The benchmark deterministically stratifies questions by `question_type` into development, validation, and held-out test partitions (70/15/15, seed 42). Development runs evaluate retrieval and reranking only; they do not generate answers. Validation runs compare complete RAG pipelines, including answer and citation quality. Standard and Agentic modes share an initial retrieval on the original question; Agentic RAG can issue additional focused searches within the configured step limit. The test partition is reserved for the final comparison. Each phase uses its corresponding split.

The hybrid benchmark compares fused BM25 + vector retrieval with reranking disabled and enabled on the development split. It does not generate answers and reports only Evidence Recall@5/10, Precision@5/10, and nDCG@10. The RRF fusion constant defaults to 60. Each query retrieves up to `candidate_depth` passages from each source, fuses by rank and deduplicates by chunk ID, then sends the fused list through the configured reranker before context selection.

Run the hybrid retrieval evaluation when the hybrid index is built and the selected vector and reranker services are available:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase hybrid --backend hybrid --split development --reranker both
```

This phase does not calculate latency, retrieval-call counts, or model-call counts. Its command is operator-invoked; implementation and CLI help checks do not run inference.

To run the Reranker Model with BGE model, run this code before:
```powershell
$modelLine = (ollama show --modelfile qllama/bge-reranker-v2-m3:latest |
    Select-String '^FROM ').Line
$modelPath = $modelLine -replace '^FROM\s+', ''

& "Path-to-your-llama\llama-server.exe" -m $modelPath --embedding --pooling rank --reranking `
    --host 127.0.0.1 --port 8080
```

Run a development sweep across vector and graph retrieval, with reranking off and on:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase tune --backend both --split development --reranker both
```

After selecting the vector backend and reranker setting, compare standard and Agentic RAG generation on the same validation questions:

```powershell
python -m agentic_ai_for_enterprise.benchmark --phase compare --backend selected --split validation --reranker selected
```

Use development retrieval metrics to narrow the candidates, then review evidence, answer, citation, latency, and model-call metrics for both generation modes on validation. Set the chosen backend and frozen reranker state in `configs/default.json`. The final command compares standard RAG with Agentic RAG on the same held-out questions:

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

The sidebar selects Vector RAG (Pinecone), Graph RAG (Neo4j), or Hybrid RAG (Pinecone + BM25). Chat uses the Agentic RAG ReAct loop with the frozen backend and reranker values from `configs/default.json`. For Hybrid RAG, each ReAct query performs BM25 and vector retrieval, RRF fusion, and then reranking. Reranker on/off is available only to benchmark runs. Responses contain `role`, `context`, and `sources`; citations link to retrieved article URLs.

## Configuration

`configs/default.json` controls the selected backend, frozen reranker state, chunking, candidate/context depths, graph traversal depth, ReAct step limit, and benchmark split. `.env` contains service connection details and model identifiers; use `.env.example` for the variable names and safe placeholders. Keep `.env` private.
