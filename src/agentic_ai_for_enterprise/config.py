"""Application and benchmark configuration loaded from files and environment."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


class ConfigurationError(ValueError):
    """Raised when configuration is missing or outside supported bounds."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    selected_backend: str = "vector"
    reranker_enabled: bool = False
    hybrid_bm25_index_path: str = "artifacts/index/hybrid_bm25.sqlite3"
    fusion_constant: int = 60
    chunk_size: int = 1000
    chunk_overlap: int = 150
    candidate_depth: int = 20
    context_depth: int = 5
    reranker_depth: int = 20
    graph_traversal_depth: int = 2
    react_max_steps: int = 4
    max_context_characters: int = 18000
    split_seed: int = 42
    development_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15

    @property
    def bm25_index_path(self) -> Path:
        path = Path(self.hybrid_bm25_index_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @classmethod
    def load(cls, path: Path = DEFAULT_CONFIG_PATH) -> PipelineConfig:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ConfigurationError(f"Cannot read pipeline config at {path}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigurationError(f"Invalid JSON in pipeline config at {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("Pipeline configuration must be a JSON object.")

        fields = cls.__dataclass_fields__
        unknown = sorted(set(raw) - set(fields))
        if unknown:
            raise ConfigurationError(f"Unknown pipeline configuration keys: {', '.join(unknown)}")
        try:
            config = cls(**raw)
        except TypeError as exc:
            raise ConfigurationError(f"Invalid pipeline configuration: {exc}") from exc
        config.validate()
        return config

    def validate(self) -> None:
        if not isinstance(self.reranker_enabled, bool):
            raise ConfigurationError("reranker_enabled must be a JSON boolean.")
        for name in (
            "chunk_size",
            "chunk_overlap",
            "candidate_depth",
            "context_depth",
            "reranker_depth",
            "graph_traversal_depth",
            "react_max_steps",
            "max_context_characters",
            "split_seed",
            "fusion_constant",
        ):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ConfigurationError(f"{name} must be an integer.")
        for name in ("development_fraction", "validation_fraction", "test_fraction"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), (int, float)):
                raise ConfigurationError(f"{name} must be a number.")
        if self.selected_backend not in {"vector", "graph", "hybrid"}:
            raise ConfigurationError("selected_backend must be 'vector', 'graph', or 'hybrid'.")
        if not isinstance(self.hybrid_bm25_index_path, str) or not self.hybrid_bm25_index_path.strip():
            raise ConfigurationError("hybrid_bm25_index_path must be a non-empty path string.")
        if self.fusion_constant < 1:
            raise ConfigurationError("fusion_constant must be at least 1.")
        if self.chunk_size < 1 or self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            raise ConfigurationError("Chunk size must be positive and overlap smaller than chunk size.")
        for name in ("candidate_depth", "context_depth", "reranker_depth", "react_max_steps", "max_context_characters"):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be at least 1.")
        if self.graph_traversal_depth < 1 or self.graph_traversal_depth > 4:
            raise ConfigurationError("graph_traversal_depth must be between 1 and 4.")
        if min(self.development_fraction, self.validation_fraction, self.test_fraction) <= 0:
            raise ConfigurationError("All benchmark split fractions must be positive.")
        if abs(self.development_fraction + self.validation_fraction + self.test_fraction - 1.0) > 1e-9:
            raise ConfigurationError("Benchmark split fractions must sum to 1.")


@dataclass(frozen=True, slots=True)
class Settings:
    ollama_base_url: str
    generation_model: str
    embedding_model: str
    reranker_model: str
    reranker_base_url: str
    pinecone_api_key: str | None
    pinecone_index_name: str | None
    pinecone_namespace: str
    pinecone_cloud: str
    pinecone_region: str
    neo4j_uri: str | None
    neo4j_username: str | None
    neo4j_password: str | None
    neo4j_database: str
    request_timeout_seconds: int = 120

    @classmethod
    def from_env(cls, *, dotenv_path: Path | None = None) -> Settings:
        """Read app settings without exporting unrelated .env values into the process."""
        file_values = dotenv_values(dotenv_path or PROJECT_ROOT / ".env")

        def value(name: str, default: str | None = None, alternate: str | None = None) -> str | None:
            if name in os.environ:
                return os.environ[name]
            if name in file_values and file_values[name] is not None:
                return str(file_values[name])
            if alternate:
                if alternate in os.environ:
                    return os.environ[alternate]
                if alternate in file_values and file_values[alternate] is not None:
                    return str(file_values[alternate])
            return default

        values: dict[str, Any] = {
            "ollama_base_url": str(value("OLLAMA_BASE_URL")).rstrip("/"),
            "generation_model": value("GENERATION_MODEL"),
            "embedding_model": value("EMBEDDING_MODEL"),
            "reranker_model": value("RERANKER_MODEL"),
            "reranker_base_url": value("RERANKER_BASE_URL"),
            "pinecone_api_key": value("PINECONE_API_KEY") or None,
            "pinecone_index_name": value("PINECONE_NAME_INDEX") or None,
            "pinecone_namespace": value("PINECONE_NAMESPACE") or None,
            "pinecone_cloud": value("PINECONE_CLOUD"),
            "pinecone_region": value("PINECONE_REGION"),
            "neo4j_uri": value("NEO4J_URI") or None,
            "neo4j_username": value("NEO4J_USERNAME", alternate="NEO4J_USER") or None,
            "neo4j_password": value("NEO4J_PASSWORD") or None,
            "neo4j_database": value("NEO4J_DATABASE", "neo4j"),
            "request_timeout_seconds": _int_value(value("REQUEST_TIMEOUT_SECONDS", "120"), "REQUEST_TIMEOUT_SECONDS"),
        }
        for name in (
            "generation_model",
            "embedding_model",
            "reranker_model",
            "reranker_base_url",
            "pinecone_cloud",
            "pinecone_region",
            "neo4j_database",
        ):
            if not values[name]:
                raise ConfigurationError(f"{name} must not be empty.")
        values["reranker_base_url"] = str(values["reranker_base_url"]).rstrip("/")
        settings = cls(**values)
        if not settings.ollama_base_url.startswith(("http://", "https://")):
            raise ConfigurationError("OLLAMA_BASE_URL must use http:// or https://.")
        if not settings.reranker_base_url.startswith(("http://", "https://")):
            raise ConfigurationError("RERANKER_BASE_URL must use http:// or https://.")
        if settings.pinecone_cloud not in {"aws", "gcp", "azure"}:
            raise ConfigurationError("PINECONE_CLOUD must be one of aws, gcp, or azure.")
        if settings.request_timeout_seconds < 1 or settings.request_timeout_seconds > 900:
            raise ConfigurationError("REQUEST_TIMEOUT_SECONDS must be between 1 and 900.")
        return settings

    def require_vector(self) -> tuple[str, str]:
        if not self.pinecone_api_key or not self.pinecone_index_name:
            raise ConfigurationError("Set PINECONE_API_KEY and PINECONE_INDEX_NAME to use Vector RAG.")
        return self.pinecone_api_key, self.pinecone_index_name

    def require_graph(self) -> tuple[str, str, str]:
        if not self.neo4j_uri or not self.neo4j_username or not self.neo4j_password:
            raise ConfigurationError("Set NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD to use Graph RAG.")
        return self.neo4j_uri, self.neo4j_username, self.neo4j_password


def _int_value(value: str | None, name: str) -> int:
    try:
        return int(value or "")
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer.") from exc
