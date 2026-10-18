# Module Spec: `vector-rag`

## Objective
Provide the vector-backed RAG variant using Ollama EmbeddingGemma and Pinecone.

## Interface
Accept traceable chunks from `corpus-ingestion`; create embeddings with `embeddinggemma:latest`; index and retrieve passages with document metadata and scores. Pass candidates to the shared `reranking` module, which may rerank or preserve their initial order based on the explicit benchmark configuration and frozen chat setting.

## Success Criteria
- Index/retrieval results retain document ID, title, and URL.
- Configuration supports tuning retrieval depth and reranking depth; benchmark runs can compare reranking enabled and disabled, while chat uses a frozen selection.
- Results satisfy the shared retrieval result contract.

## Boundaries
- Always record embedding and reranker model identifiers.
- Ask first before changing the Pinecone service/index design.
- Never expose Pinecone credentials in source or run artifacts.

## Implementation choices
The index is serverless Pinecone, configured by environment variables, uses the configured namespace, cosine similarity, and validates its dimension against an Ollama embedding probe. The shared reranker uses the configured Ollama Qwen3 reranker.
