"""Execute an explicit, bounded ReAct prompt and retrieval-tool loop."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from pydantic import BaseModel, ConfigDict, Field

from ..config import PipelineConfig, Settings
from ..contracts import RagAnswer, RetrievedPassage, Retriever
from ..response.answer import generate_answer
from ..response.validation import insufficient_evidence
from .prompts import REACT_SYSTEM_PROMPT

# Uses LangChain tools with an explicit ReAct loop: https://docs.langchain.com/oss/python/langchain/agents

LOGGER = logging.getLogger(__name__)


class RetrievalToolInput(BaseModel):
    query: str = Field(min_length=1, max_length=1000)


class FinalAgentAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    answer: str = Field(min_length=1, max_length=10000)
    source_chunk_ids: list[str] = Field(default_factory=list, max_length=20)


@dataclass(frozen=True, slots=True)
class AgentRun:
    answer: RagAnswer
    passages: tuple[RetrievedPassage, ...]
    retrieval_calls: int
    model_calls: int
    elapsed_seconds: float
    retrieval_queries: tuple[str, ...]


def answer_with_react(
    question: str,
    retriever: Retriever,
    settings: Settings,
    pipeline: PipelineConfig,
) -> AgentRun:
    started = time.perf_counter()
    seen: dict[str, RetrievedPassage] = {}
    retrieval_queries: list[str] = []
    observations: list[str] = []
    retrieval_calls = 0
    model_calls = 0
    llm = ChatOllama(
        model=settings.generation_model,
        base_url=settings.ollama_base_url,
        temperature=0,
        num_predict=512,
        timeout=settings.request_timeout_seconds,
    )

    @tool("retrieve_internal_evidence", args_schema=RetrievalToolInput)
    def retrieve_internal_evidence(query: str) -> str:
        """Search the configured internal document retriever for relevant passages."""
        nonlocal retrieval_calls
        query = query.strip()
        if not query or len(query) > 1000:
            return "Invalid retrieval query. Provide 1 to 1000 characters."
        retrieval_calls += 1
        retrieval_queries.append(query)
        passages = retriever.retrieve(query, pipeline.candidate_depth)
        compact: list[dict[str, str | float]] = []
        for passage in passages[: pipeline.context_depth]:
            current = seen.get(passage.chunk_id)
            if current is None or passage.score > current.score:
                seen[passage.chunk_id] = passage
            compact.append(
                {
                    "chunk_id": passage.chunk_id,
                    "title": passage.title,
                    "url": passage.url,
                    "text": passage.text[:3500],
                    "score": passage.score,
                }
            )
        return json.dumps(compact, ensure_ascii=False)

    tool_map = {retrieve_internal_evidence.name: retrieve_internal_evidence}
    prompt = REACT_SYSTEM_PROMPT.replace("{max_steps}", str(pipeline.react_max_steps))
    for _ in range(pipeline.react_max_steps):
        history = "\n\n".join(observations)
        response = llm.invoke(
            [
                ("system", prompt),
                (
                    "human",
                    f"Question:\n{question}\n\nPrevious actions and observations:\n"
                    f"{history or '(none yet)'}\n\nContinue with exactly one Action or a Final Answer.",
                ),
            ]
        )
        model_calls += 1
        text = _content_text(response.content)
        final = _parse_final_answer(text)
        if final is not None:
            answer = _answer_from_final(final, seen)
            return AgentRun(
                answer=answer,
                passages=tuple(seen.values()),
                retrieval_calls=retrieval_calls,
                model_calls=model_calls,
                elapsed_seconds=time.perf_counter() - started,
                retrieval_queries=tuple(retrieval_queries),
            )

        action = _parse_action(text)
        if action is None:
            observations.append("Observation: Invalid format. Use the documented retrieval action or final JSON format.")
            continue
        action_name, action_query = action
        action_tool = tool_map.get(action_name)
        if action_tool is None:
            observations.append("Observation: Unknown action. The only allowed action is retrieve_internal_evidence.")
            continue
        try:
            observation = action_tool.invoke({"query": action_query})
        except Exception as exc:
            LOGGER.warning("ReAct retrieval action failed.")
            observations.append("Observation: Retrieval failed. Try a shorter, focused query or provide an insufficient-evidence answer.")
            continue
        observations.append(f"Action: {action_name}\nAction Input: {json.dumps({'query': action_query})}\nObservation: {observation}")

    if seen:
        ordered = list(seen.values())
        fallback = generate_answer(question, ordered, settings, pipeline)
        model_calls += 1
    else:
        fallback = insufficient_evidence()
    return AgentRun(
        answer=fallback,
        passages=tuple(seen.values()),
        retrieval_calls=retrieval_calls,
        model_calls=model_calls,
        elapsed_seconds=time.perf_counter() - started,
        retrieval_queries=tuple(retrieval_queries),
    )


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = [item.get("text", "") for item in content if isinstance(item, dict)]
        return "".join(str(part) for part in parts)
    return str(content)


def _parse_action(text: str) -> tuple[str, str] | None:
    action_match = re.search(r"^Action:\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*$", text, re.MULTILINE)
    input_match = re.search(r"^Action Input:\s*(.+?)\s*$", text, re.MULTILINE)
    if not action_match or not input_match:
        return None
    raw_input = input_match.group(1).strip()
    try:
        parsed = json.loads(raw_input)
    except json.JSONDecodeError:
        query = raw_input.strip('"')
    else:
        query = parsed.get("query", "") if isinstance(parsed, dict) else ""
    query = query.strip() if isinstance(query, str) else ""
    if not query or len(query) > 1000:
        return None
    return action_match.group(1), query


def _parse_final_answer(text: str) -> FinalAgentAnswer | None:
    marker = re.search(r"^Final Answer:\s*", text, re.MULTILINE)
    if not marker:
        return None
    payload = text[marker.end() :].strip()
    if payload.startswith("```"):
        payload = re.sub(r"^```(?:json)?\s*|\s*```$", "", payload, flags=re.IGNORECASE)
    try:
        value, _ = json.JSONDecoder().raw_decode(payload)
        return FinalAgentAnswer.model_validate(value)
    except Exception:
        return None


def _answer_from_final(final: FinalAgentAnswer, seen: dict[str, RetrievedPassage]) -> RagAnswer:
    if not final.source_chunk_ids:
        return insufficient_evidence()
    urls: list[str] = []
    for chunk_id in final.source_chunk_ids:
        passage = seen.get(chunk_id)
        if passage is None:
            return insufficient_evidence()
        urls.append(passage.url)
    return RagAnswer(role="assistant", context=final.answer, sources=list(dict.fromkeys(urls)))
