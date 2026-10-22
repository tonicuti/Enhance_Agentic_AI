# Capability Map: RAG Retrieval Evaluation and ReAct Integration

| Module ID | Responsibility | Depends on |
|---|---|---|
| `corpus-ingestion` | Validate article records, normalize metadata, and create traceable text chunks. | None |
| `vector-rag` | Embed chunks with Ollama EmbeddingGemma and index/retrieve them in Pinecone. | `corpus-ingestion` |
| `bm25-rag` | Index the same traceable chunks in a local SQLite FTS5 index and retrieve lexical candidates by BM25 rank. | `corpus-ingestion` |
| `hybrid-retrieval` | Retrieve vector and BM25 candidate lists and fuse them with RRF before reranking. | `vector-rag`, `bm25-rag` |
| `graph-rag` | Extract entities and relations with Ollama Qwen3, preserve provenance, and retrieve through Neo4j. | `corpus-ingestion` |
| `reranking` | Optionally rerank candidates with the configured BGE reranker; honor explicit enabled/disabled settings. | `vector-rag`, `graph-rag`, `hybrid-retrieval` |
| `rag-benchmark` | Tune and compare retrievers and reranker modes using MultiHopRAG; score evidence and answers, with runtime metrics omitted for hybrid retrieval evaluation. | `vector-rag`, `graph-rag`, `hybrid-retrieval`, `reranking` |
| `react-retrieval` | Expose the selected retriever to a bounded LangChain ReAct loop, preserving the benchmark-selected reranker configuration. | `rag-benchmark`, `reranking` |
| `response-format` | Validate and return grounded answers as `role`, `context`, and URL-list `sources`. | Selected pipeline; `react-retrieval` for Agentic RAG |
| `chat-interface` | Streamlit chat that selects a retrieval backend and renders grounded answers and source links; reranker settings remain fixed. | `response-format`, selected pipeline |

Build order: `corpus-ingestion` -> `vector-rag`, `bm25-rag`, and `graph-rag` -> `reranking` and `hybrid-retrieval` -> `rag-benchmark` -> select and tune a retrieval pipeline -> `react-retrieval` -> `response-format` -> `chat-interface`.

The vector and graph retrievers remain independent alternatives; the hybrid retriever combines BM25 and vector search. ReAct is the retrieval orchestration layer and can use any selected backend, including Hybrid RAG. The final benchmark compares standard RAG with its ReAct-based Agentic RAG counterpart on the same held-out questions.
