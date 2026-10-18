# Module Spec: `reranking`

## Objective
Provide one shared optional reranking stage for vector and graph candidate passages using `dengcao/Qwen3-Reranker-0.6B:Q8_0` through Ollama.

## Interface
Accept ranked candidates and an explicit `reranker_enabled: bool` setting from benchmark configuration. When enabled, rerank candidates with the configured Ollama model. When disabled, return candidates in their original order. Preserve document ID, title, URL, passage text, and relevant score metadata. The benchmark may run both settings to compare retrieval quality; once selected, the setting is frozen for the chat and ReAct pipelines and cannot be changed per chat request.

## Success Criteria
- Both backend variants use the same reranking behavior and configuration.
- The enabled/disabled state is recorded for benchmark runs and the selected state is propagated consistently to standard and ReAct retrieval.
- Disabled mode does not invoke the reranker model.

## Boundaries
- Always make the setting explicit in benchmark runs; do not expose a per-request reranker control in the chat UI.
- Ask first before changing the reranker model or introducing backend-specific reranking.
- Never lose source provenance during reranking.

## Implementation choices
The LangChain `ChatOllama` integration sends the Qwen3 yes/no relevance prompt and uses available yes/no token probabilities, with a validated yes/no text fallback. Benchmark selection is recorded in `configs/default.json`; reranker enabled/disabled comparisons remain evaluation-only.
