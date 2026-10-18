# Module Spec: `react-retrieval`

## Objective
Expose the selected standard RAG retriever as a LangChain tool and enable iterative retrieval with a ReAct prompt.

## Interface
The agent receives a question and uses the benchmark-selected, frozen reranker configuration when retrieving ranked evidence. It inspects the returned passages and may issue refined retrieval queries up to a configurable maximum. Each retrieval call honors the same setting. It returns an evidence-grounded answer and the URLs used. The chat UI cannot override the setting per request.

## Success Criteria
- The agent can make bounded retrieval calls and terminate with a final answer or an insufficient-evidence response.
- ReAct output can be benchmarked against the selected standard RAG system on the same test questions.
- Retrieval actions and results are recorded for analysis.

## Boundaries
- Always preserve provenance and enforce a maximum number of retrieval steps.
- Ask first before changing the selected retriever or prompt strategy after final benchmark configuration is frozen.
- Never cite a URL that was not returned by the retrieval tool.

## Implementation choices
The explicit prompt uses Thought/Action/Action Input/Observation/Final Answer format. The default maximum is four retrieval actions. Final citations use chunk IDs observed from the tool, then map to verified article URLs. If the loop ends without a valid final answer, the system generates a grounded fallback from accumulated evidence or returns an insufficient-evidence answer.
