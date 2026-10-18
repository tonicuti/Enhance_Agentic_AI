"""Validated graph extraction records with mandatory source provenance."""

from __future__ import annotations

import hashlib
import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ExtractedEntity(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    name: str = Field(min_length=1, max_length=200)
    entity_type: str = Field(min_length=1, max_length=80)


class ExtractedRelation(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str = Field(min_length=1, max_length=200)
    predicate: str = Field(min_length=1, max_length=120)
    object: str = Field(min_length=1, max_length=200)
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)


class GraphExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[ExtractedEntity] = Field(default_factory=list, max_length=100)
    relations: list[ExtractedRelation] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def relation_endpoints_exist(self) -> Self:
        names = {normalize_entity_name(item.name) for item in self.entities}
        for relation in self.relations:
            if normalize_entity_name(relation.subject) not in names:
                raise ValueError("relation subject must be included in entities")
            if normalize_entity_name(relation.object) not in names:
                raise ValueError("relation object must be included in entities")
        return self


class GraphRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    url: str = Field(min_length=1)
    entities: list[ExtractedEntity]
    relations: list[ExtractedRelation]


def normalize_entity_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip()).casefold()


def entity_id(name: str, entity_type: str) -> str:
    key = f"{normalize_entity_name(name)}\0{entity_type.strip().casefold()}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def relation_id(chunk_id: str, relation: ExtractedRelation) -> str:
    key = "\0".join(
        (
            chunk_id,
            normalize_entity_name(relation.subject),
            relation.predicate.strip().casefold(),
            normalize_entity_name(relation.object),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()
