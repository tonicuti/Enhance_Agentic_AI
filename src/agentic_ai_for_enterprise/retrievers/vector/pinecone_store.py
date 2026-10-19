"""Pinecone-backed retrieval through LangChain's vector store integration."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

from ...config import Settings
from ...contracts import RetrievedPassage, TextChunk

# Integration API: https://docs.langchain.com/oss/python/integrations/vectorstores/pinecone
# Ollama embeddings API: https://docs.langchain.com/oss/python/integrations/embeddings/ollama

class PineconeRetriever:
    def __init__(self, settings: Settings) -> None:
        api_key, index_name = settings.require_vector()
        self.settings = settings
        self.index_name = index_name
        self.client = Pinecone(api_key=api_key)
        self.embeddings = OllamaEmbeddings(
            model=settings.embedding_model,
            base_url=settings.ollama_base_url,
        )
        self.vector_store: PineconeVectorStore | None = None

    def ensure_index(self, *, create_if_missing: bool = False) -> None:
        """Create the configured serverless index from the live embedding size if absent."""
        dimension = len(self.embeddings.embed_query("embedding dimension probe"))
        if not self.client.has_index(self.index_name):
            if not create_if_missing:
                raise ValueError(f"Pinecone index '{self.index_name}' does not exist. Run vector indexing first.")
            self.client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=self.settings.pinecone_cloud,
                    region=self.settings.pinecone_region,
                ),
            )
        description = self.client.describe_index(self.index_name)
        if description.dimension != dimension:
            raise ValueError(
                f"Pinecone index '{self.index_name}' has dimension {description.dimension}; "
                f"{self.settings.embedding_model} produced {dimension}. Use a matching index."
            )
        if description.metric != "cosine":
            raise ValueError(
                f"Pinecone index '{self.index_name}' uses metric '{description.metric}'; "
                "this pipeline requires cosine similarity."
            )
        self.vector_store = PineconeVectorStore(
            index=self.client.Index(self.index_name),
            embedding=self.embeddings,
            namespace=self.settings.pinecone_namespace,
        )

    def index(self, chunks: list[TextChunk]) -> int:
        if not chunks:
            return 0
        self.ensure_index(create_if_missing=True)
        documents = [Document(page_content=chunk.text, metadata=_chunk_metadata(chunk)) for chunk in chunks]
        assert self.vector_store is not None
        self.vector_store.add_documents(documents=documents, ids=[chunk.chunk_id for chunk in chunks], embedding_chunk_size = 32)
        return len(chunks)

    def retrieve(self, query: str, limit: int) -> list[RetrievedPassage]:
        if not query.strip():
            return []
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if self.vector_store is None:
            self.ensure_index()
        assert self.vector_store is not None
        matches = self.vector_store.similarity_search_with_score(query, k=limit)
        return [
            RetrievedPassage(
                document_id=str(doc.metadata["document_id"]),
                chunk_id=str(doc.metadata["chunk_id"]),
                title=str(doc.metadata["title"]),
                url=str(doc.metadata["url"]),
                text=doc.page_content,
                score=float(score),
                source=_metadata_string(doc.metadata.get("source")),
                category=_metadata_string(doc.metadata.get("category")),
                chunk_index=int(doc.metadata.get("chunk_index", 0)),
                score_type="cosine_similarity",
            )
            for doc, score in matches
        ]


def _chunk_metadata(chunk: TextChunk) -> dict[str, Any]:
    return {
        "chunk_id": chunk.chunk_id,
        "document_id": chunk.document_id,
        "title": chunk.title,
        "url": chunk.url,
        "source": chunk.source or "",
        "category": chunk.category or "",
        "published_at": chunk.published_at or "",
        "chunk_index": chunk.chunk_index,
    }


def _metadata_string(value: object) -> str | None:
    return str(value) if value not in (None, "") else None
