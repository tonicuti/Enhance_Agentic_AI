"""Read and validate article records from the internal corpus JSON file."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from ..contracts import Article


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    record_index: int
    reason: str


def canonicalize_url(value: str) -> str:
    """Normalize a URL while retaining query parameters that may identify content."""
    parsed = urlsplit(value.strip())
    scheme = parsed.scheme.lower()
    host = (parsed.hostname or "").lower()
    if scheme not in {"http", "https"} or not host:
        raise ValueError("url must be an absolute http(s) URL")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("url contains an invalid port") from exc
    if parsed.username or parsed.password:
        raise ValueError("url must not contain embedded credentials")
    default_port = (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    netloc = host if port is None or default_port else f"{host}:{port}"
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, query, ""))


def document_id_for_url(url: str) -> str:
    canonical = canonicalize_url(url)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_corpus(path: Path) -> tuple[list[Article], list[ValidationIssue]]:
    """Load valid article records and return diagnostics for every invalid row."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"Cannot read corpus file at {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Corpus file is not valid JSON at line {exc.lineno}: {exc.msg}") from exc
    if not isinstance(raw, list):
        raise ValueError("Corpus JSON root must be an array of article objects.")

    articles: list[Article] = []
    issues: list[ValidationIssue] = []
    seen_ids: set[str] = set()
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            issues.append(ValidationIssue(index, "record must be a JSON object"))
            continue
        required = ("title", "url", "body")
        missing = [key for key in required if not isinstance(record.get(key), str) or not record[key].strip()]
        if missing:
            issues.append(ValidationIssue(index, f"missing or empty required fields: {', '.join(missing)}"))
            continue
        try:
            canonical_url = canonicalize_url(record["url"])
        except ValueError as exc:
            issues.append(ValidationIssue(index, str(exc)))
            continue
        document_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()
        if document_id in seen_ids:
            issues.append(ValidationIssue(index, "duplicate canonical URL; earlier record is retained"))
            continue
        seen_ids.add(document_id)
        articles.append(
            Article(
                document_id=document_id,
                title=record["title"].strip(),
                url=canonical_url,
                body=record["body"].strip(),
                source=_optional_string(record.get("source")),
                category=_optional_string(record.get("category")),
                published_at=_optional_string(record.get("published_at")),
            )
        )
    return articles, issues


def _optional_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
