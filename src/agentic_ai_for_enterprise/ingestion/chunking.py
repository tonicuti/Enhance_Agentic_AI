"""Create reproducible LangChain text chunks with source metadata."""

from __future__ import annotations

import hashlib

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..config import PipelineConfig
from ..contracts import Article, TextChunk


def chunk_articles(articles: list[Article], config: PipelineConfig) -> list[TextChunk]:
    config.validate()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        add_start_index=True,
    )
    chunks: list[TextChunk] = []
    for article in articles:
        pieces = splitter.split_text(article.body)
        for chunk_index, text in enumerate(pieces):
            chunk_key = f"{article.document_id}:{chunk_index}"
            chunk_id = hashlib.sha256(chunk_key.encode("utf-8")).hexdigest()
            chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    document_id=article.document_id,
                    title=article.title,
                    url=article.url,
                    text=text,
                    chunk_index=chunk_index,
                    source=article.source,
                    category=article.category,
                    published_at=article.published_at,
                )
            )
    return chunks
