"""Streamlit chat interface for the internal document RAG system."""

import streamlit as st


st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="📚",
    layout="wide",
)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("Knowledge Assistant")
st.caption("Ask questions across the indexed document collection.")

with st.sidebar:
    st.header("Retrieval settings")

    backend = st.selectbox(
        "RAG system",
        options=["Vector RAG · Pinecone", "Graph RAG · Neo4j"],
        key="retrieval_backend",
        help="Choose which retrieval pipeline to use once the backend is connected.",
    )

    st.divider()
    st.warning(
        "The chat interface is ready, but retrieval and answer generation are not "
        "connected yet. Submitted questions will show a setup notice instead of a "
        "document answer."
    )

    # The current Streamlit button API uses width="stretch" for full width.
    # Source: https://docs.streamlit.io/1.64.0/develop/api-reference/widgets/st.button
    if st.button("Clear conversation", width="stretch"):
        st.session_state.messages = []


def render_message(message: dict[str, object]) -> None:
    """Render one stored chat message and any source links it contains."""
    role = str(message["role"])
    with st.chat_message(role):
        st.text(str(message["content"]))
        if role == "assistant" and "backend" in message:
            st.caption(str(message["backend"]))
        sources = message.get("sources", [])
        if isinstance(sources, list) and sources:
            st.caption("Sources")
            for source_url in sources:
                # Render model-provided citations as plain text, not Markdown.
                st.text(str(source_url))


# Streamlit chat messages and input provide the conversational layout.
# Source: https://docs.streamlit.io/develop/api-reference/chat
for stored_message in st.session_state.messages:
    render_message(stored_message)

if not st.session_state.messages:
    st.info("Start with a question about the documents in the corpus.")

prompt = st.chat_input("Ask a question about the documents", max_chars=4000)
if prompt:
    user_message = {"role": "user", "content": prompt}
    assistant_message = {
        "role": "assistant",
        "content": (
            "The retrieval pipeline is not connected yet, so I can’t answer from "
            "the documents. The selected backend is ready for integration."
        ),
        "backend": backend,
        "sources": [],
    }
    st.session_state.messages.extend([user_message, assistant_message])
    render_message(user_message)
    render_message(assistant_message)
