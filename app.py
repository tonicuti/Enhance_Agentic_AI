"""Streamlit chat interface for the internal document RAG system."""

from __future__ import annotations

import logging
from urllib.parse import urlsplit

import streamlit as st

from agentic_ai_for_enterprise.application import RagApplication
from agentic_ai_for_enterprise.config import ConfigurationError, PipelineConfig, Settings

# Streamlit chat components and link buttons: https://docs.streamlit.io/develop/api-reference/chat
# https://docs.streamlit.io/develop/api-reference/widgets/st.link_button

LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")


@st.cache_resource(show_spinner=False)
def _application_for_backend(backend: str) -> RagApplication:
    """Reuse configured service clients across Streamlit reruns."""
    return RagApplication(
        backend=backend,
        settings=Settings.from_env(),
        pipeline=PipelineConfig.load(),
    )


def _render_message(message: dict[str, object], message_index: int) -> None:
    role = str(message.get("role", "assistant"))
    with st.chat_message(role):
        st.write(str(message.get("context", "")))
        sources = message.get("sources", [])
        if isinstance(sources, list) and sources:
            st.caption("Sources")
            for source_index, url in enumerate(sources):
                parts = urlsplit(str(url))
                st.link_button(
                    parts.netloc or "Source article",
                    str(url),
                    key=f"source-{message_index}-{source_index}",
                )


st.title("Knowledge Assistant")
st.caption("Ask questions across the indexed document collection.")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Retrieval settings")
    backend_label = st.selectbox(
        "RAG system",
        options=["Vector RAG · Pinecone", "Graph RAG · Neo4j", "Hybrid RAG · Pinecone + BM25"],
        key="retrieval_backend",
        help="Choose the internal document retrieval backend.",
    )
    backend = (
        "vector"
        if backend_label.startswith("Vector")
        else "graph"
        if backend_label.startswith("Graph")
        else "hybrid"
    )
    st.divider()
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []

for message_index, stored_message in enumerate(st.session_state.messages):
    _render_message(stored_message, message_index)

if not st.session_state.messages:
    st.info("Start with a question about the documents in the corpus.")

prompt = st.chat_input("Ask a question about the documents", max_chars=4000)
if prompt:
    st.session_state.messages.append({"role": "user", "context": prompt})
    _render_message(st.session_state.messages[-1], len(st.session_state.messages) - 1)
    with st.chat_message("assistant"):
        try:
            with st.status("Searching the indexed documents…", expanded=False):
                result = _application_for_backend(backend).answer(prompt, mode="agentic")
            assistant_message = result.answer.to_dict()
        except ConfigurationError as exc:
            assistant_message = {"role": "assistant", "context": str(exc), "sources": []}
        except Exception as exc:
            LOGGER.warning("RAG request failed for backend %s (%s).", backend, type(exc).__name__)
            assistant_message = {
                "role": "assistant",
                "context": (
                    "Retrieval is temporarily unavailable. Check that the selected "
                    "Ollama and database services are configured and running."
                ),
                "sources": [],
            }
        st.write(assistant_message["context"])
        sources = assistant_message.get("sources", [])
        if sources:
            st.caption("Sources")
            for source_index, url in enumerate(sources):
                parts = urlsplit(str(url))
                st.link_button(
                    parts.netloc or "Source article",
                    str(url),
                    key=f"source-{len(st.session_state.messages)}-{source_index}",
                )
    st.session_state.messages.append(assistant_message)
