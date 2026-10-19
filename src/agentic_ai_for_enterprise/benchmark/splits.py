"""Deterministic stratified development, validation, and test split assignment."""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass

from ..config import PipelineConfig
from ..contracts import BenchmarkQuestion


@dataclass(frozen=True, slots=True)
class DatasetSplits:
    development: tuple[BenchmarkQuestion, ...]
    validation: tuple[BenchmarkQuestion, ...]
    test: tuple[BenchmarkQuestion, ...]
    membership: dict[str, str]
    fingerprint: str
    seed: int


def split_questions(questions: list[BenchmarkQuestion], config: PipelineConfig) -> DatasetSplits:
    config.validate()
    groups: dict[str, list[BenchmarkQuestion]] = defaultdict(list)
    for question in questions:
        groups[question.question_type].append(question)

    assigned: dict[str, list[BenchmarkQuestion]] = {"development": [], "validation": [], "test": []}
    membership: dict[str, str] = {}
    fractions = (config.development_fraction, config.validation_fraction, config.test_fraction)
    split_names = tuple(assigned)
    for question_type in sorted(groups):
        group = sorted(groups[question_type], key=lambda item: item.question_id)
        random.Random(f"{config.split_seed}:{question_type}").shuffle(group)
        counts = _allocate_counts(len(group), fractions)
        start = 0
        for split_name, count in zip(split_names, counts):
            portion = group[start : start + count]
            assigned[split_name].extend(portion)
            membership.update({question.question_id: split_name for question in portion})
            start += count

    for split in assigned.values():
        split.sort(key=lambda item: item.question_id)
    split_payload = (
        f"{config.split_seed}:{config.development_fraction}:{config.validation_fraction}:"
        f"{config.test_fraction}:" + ",".join(
            f"{key}={membership[key]}" for key in sorted(membership)
        )
    )
    split_fingerprint = hashlib.sha256(split_payload.encode("utf-8")).hexdigest()
    return DatasetSplits(
        development=tuple(assigned["development"]),
        validation=tuple(assigned["validation"]),
        test=tuple(assigned["test"]),
        membership=membership,
        fingerprint=split_fingerprint,
        seed=config.split_seed,
    )


def _allocate_counts(size: int, fractions: tuple[float, float, float]) -> tuple[int, int, int]:
    raw = [size * fraction for fraction in fractions]
    counts = [int(value) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(3), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts[0], counts[1], counts[2]
