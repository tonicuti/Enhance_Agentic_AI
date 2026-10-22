"""Index corpus data into the independently selectable retrieval backends."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, PipelineConfig, Settings
from .ingestion import chunk_articles, load_corpus


LOGGER = logging.getLogger("rag_system.index")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("vector", "graph", "hybrid", "both"), required=True)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "corpus.json")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    return parser


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = build_parser().parse_args()
    pipeline = PipelineConfig.load(args.config)
    settings = Settings.from_env()
    articles, issues = load_corpus(args.corpus)
    for issue in issues:
        LOGGER.warning("Corpus record %d: %s", issue.record_index, issue.reason)
    if not articles:
        raise SystemExit("No valid corpus articles were found; nothing was indexed.")

    chunks = chunk_articles(articles, pipeline)
    LOGGER.info("Prepared %d articles and %d chunks (%d record issue(s)).", len(articles), len(chunks), len(issues))
    backends = ("vector", "graph") if args.backend == "both" else (args.backend,)
    for backend in backends:
        if backend == "vector":
            from .retrievers.vector import PineconeRetriever

            indexed = PineconeRetriever(settings).index(chunks)
        elif backend == "graph":
            from .retrievers.graph import Neo4jGraphStore
            from .retrievers.graph.extraction import extract_graph_records

            store = Neo4jGraphStore(settings)
            try:
                store.ensure_schema()
                graph_records = extract_graph_records(chunks, settings)
                indexed_chunk_ids = {record.chunk_id for record in graph_records}
                indexable_chunks = [chunk for chunk in chunks if chunk.chunk_id in indexed_chunk_ids]
                if not indexable_chunks:
                    raise RuntimeError("No chunks have valid graph extractions; nothing was indexed.")
                indexed = store.index(indexable_chunks, graph_records)
            finally:
                store.close()
        else:
            from .retrievers.bm25 import SQLiteBM25Store
            from .retrievers.vector import PineconeRetriever

            vector_indexed = PineconeRetriever(settings).index(chunks)
            bm25_store = SQLiteBM25Store(pipeline.bm25_index_path)
            try:
                bm25_indexed = bm25_store.index(chunks)
            finally:
                bm25_store.close()
            LOGGER.info(
                "Indexed %d chunks into Pinecone and %d chunks into the local BM25 index.",
                vector_indexed,
                bm25_indexed,
            )
            continue
        LOGGER.info("Indexed %d chunks into the %s backend.", indexed, backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
