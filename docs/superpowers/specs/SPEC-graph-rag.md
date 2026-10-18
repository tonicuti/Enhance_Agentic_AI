# Module Spec: `graph-rag`

## Objective
Create a graph-backed RAG variant by extracting article entities and relations with Ollama and storing provenance-linked graph records in Neo4j.

## Interface
Accept corpus documents/chunks; use `qwen3:1.7b` for entity/relation extraction; persist source article and chunk provenance; resolve query entities and traverse Neo4j to return ranked evidence candidates; pass candidates to the shared optional `reranking` module.

## Success Criteria
- Every returned passage maps to a source article URL.
- Stored entities/relations pass schema validation and include provenance.
- Retrieval results satisfy the shared retrieval result contract.
- The graph candidate order is preserved when reranking is disabled.

## Boundaries
- Always construct the graph from corpus content only.
- Ask first before adding vector search inside the graph variant, since that changes the comparison.
- Never ingest gold evidence labels into Neo4j.

## Implementation choices
The Neo4j graph stores `Article`, `Chunk`, and `Entity` nodes with provenance-linked `HAS_CHUNK`, `MENTIONS`, and `RELATED` relationships. Entity IDs are stable hashes of normalized name and type; relation confidence contributes to candidate ranking. Query traversal is bounded to the configured depth (1-4, default 2). Neo4j URI, database, and credentials are supplied through environment variables.
