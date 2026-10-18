# Capability Map: RAG Retrieval Evaluation and ReAct Integration

| Module ID | Responsibility | Depends on |
|---|---|---|
| `corpus-ingestion` | Validate article records, normalize metadata, and create traceable text chunks. | None |
| `vector-rag` | Embed chunks with Ollama EmbeddingGemma and index/retrieve them in Pinecone. | `corpus-ingestion` |
| `graph-rag` | Extract entities and relations with Ollama Qwen3, preserve provenance, and retrieve through Neo4j. | `corpus-ingestion` |
| `reranking` | Optionally rerank candidates with the configured Ollama reranker; honor explicit enabled/disabled settings. | `vector-rag`, `graph-rag` |
| `rag-benchmark` | Tune and compare retrievers and reranker modes using MultiHopRAG; score evidence, answers, citations, and efficiency. | `vector-rag`, `graph-rag`, `reranking` |
| `react-retrieval` | Expose the selected retriever to a bounded LangChain ReAct loop, preserving the benchmark-selected reranker configuration. | `rag-benchmark`, `reranking` |
| `response-format` | Validate and return grounded answers as `role`, `context`, and URL-list `sources`. | Selected pipeline; `react-retrieval` for Agentic RAG |
| `chat-interface` | Streamlit chat that selects a retrieval backend and renders grounded answers and source links; reranker settings remain fixed. | `response-format`, selected pipeline |

Build order: `corpus-ingestion` -> `vector-rag` and `graph-rag` -> `reranking` -> `rag-benchmark` -> select and tune a retrieval pipeline -> `react-retrieval` -> `response-format` -> `chat-interface`.

The vector and graph retrievers are evaluated separately. ReAct is introduced after selecting the strongest standard RAG configuration. The final benchmark compares that standard RAG configuration with its ReAct-based Agentic RAG counterpart on the same held-out questions.
