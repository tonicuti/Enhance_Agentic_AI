# Capability Map: RAG Retrieval Evaluation and ReAct Integration

| Module ID | Responsibility | Depends on |
|---|---|---|
| `corpus-ingestion` | Validate article records, normalize metadata, and create traceable text chunks. | — |
| `vector-rag` | Embed chunks with Ollama EmbeddingGemma and index/retrieve them in Pinecone. | `corpus-ingestion` |
| `graph-rag` | Extract entities and relations with Ollama Qwen3, preserve provenance, and retrieve through Neo4j. | `corpus-ingestion` |
| `reranking` | Optionally rerank candidate passages with the configured Ollama reranker; honor an explicit enabled/disabled setting. | `vector-rag`, `graph-rag` |
| `rag-benchmark` | Tune and compare retrievers and reranker modes using MultiHopRAG; score evidence, answers, citations, and efficiency. | `vector-rag`, `graph-rag`, `reranking` |
| `react-retrieval` | Expose the selected retriever to a LangChain ReAct agent for iterative retrieval, preserving the benchmark-selected reranker configuration. | `rag-benchmark`, `reranking` |
| `response-format` | Validate and return grounded answers as `role`, `context`, and URL-list `sources`. | Selected pipeline; `react-retrieval` for Agentic RAG |
| `chat-interface` | Streamlit chat page to accept questions and show grounded responses; it uses the selected pipeline configuration without exposing reranker controls. | `response-format`, selected pipeline |

Build order: `corpus-ingestion` → `vector-rag` and `graph-rag` → `reranking` → `rag-benchmark` → select and tune a retrieval pipeline → `react-retrieval` → `response-format` → `chat-interface`.

The vector and graph retrievers are evaluated separately. ReAct is introduced after selecting the strongest standard RAG configuration. The final benchmark compares that standard RAG configuration with its ReAct-based Agentic RAG counterpart on the same held-out questions.
