# Module Spec: `vector-rag`

## Objective
Provide the vector-backed RAG variant using Ollama EmbeddingGemma and Pinecone.

## Interface
Accept traceable chunks from `corpus-ingestion`; create embeddings with `embeddinggemma:latest`; index and retrieve passages with document metadata and scores. Pass candidates to the shared `reranking` module, which may rerank or preserve their initial order based on the request setting.

## Success Criteria
- Index/retrieval results retain document ID, title, and URL.
- Configuration supports tuning retrieval depth and reranking depth; reranking can be enabled or disabled per run/request.
- Results satisfy the shared retrieval result contract.

## Boundaries
- Always record embedding and reranker model identifiers.
- Ask first before changing the Pinecone service/index design.
- Never expose Pinecone credentials in source or run artifacts.

## Open Decisions
Pinecone index configuration, namespace policy, distance metric, and reranker invocation method.
