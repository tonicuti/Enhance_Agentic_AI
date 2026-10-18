# Module Spec: `chat-interface`

## Objective
Provide a Streamlit user interface for asking corpus questions and viewing grounded answers and sources.

## Interface
The user can submit a question, select Vector RAG (Pinecone) or Graph RAG (Neo4j), and view the answer plus article URLs. The UI uses the selected pipeline's frozen configuration and does not expose a reranker control; reranker on/off comparisons are performed by the benchmark. Until a retrieval service is implemented, the UI must show a clear setup notice and must not fabricate an answer. The first implementation is `app.py` using Streamlit 1.59 or newer (below 2.0).

## Success Criteria
- The chat UI does not let users change reranking state per request.
- The selected pipeline configuration is used consistently for chat and its retrieval runs.
- Responses render `context` and the URL list from `sources`.
- The UI retains conversation messages for the current Streamlit session.

## Boundaries
- Always use the benchmark-selected reranking setting across selected retrieval paths.
- Ask first before adding login or deployment requirements.
- Never imply reranking ran when the setting was disabled or the model call failed.

## Open Decisions
Deployment target.
