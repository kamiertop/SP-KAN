"""Deterministic dataset-list splitting helpers used by training."""

from __future__ import annotations

import random
from collections.abc import Sequence


def split_train_validation(
    entries: Sequence[str], val_ratio: float = 0.1, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Split a training list into disjoint, reproducible train/validation lists.

    The source sequence is never modified.  For a non-empty dataset with at
    least two samples, validation contains at least one and train retains at
    least one sample.
    """
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1")
    entries = list(entries)
    if len(entries) < 2:
        return entries, []
    val_count = max(1, int(len(entries) * val_ratio))
    val_count = min(val_count, len(entries) - 1)
    val_indices = set(random.Random(seed).sample(range(len(entries)), val_count))
    train = [item for index, item in enumerate(entries) if index not in val_indices]
    validation = [item for index, item in enumerate(entries) if index in val_indices]
    return train, validation
