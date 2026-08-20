"""Aggregation configuration for TriX-EEG level scores."""

from __future__ import annotations

LEGACY_AGGREGATION = "legacy_arithmetic"
AGGREGATION_CHOICES = (LEGACY_AGGREGATION,)


def validate_aggregation(name: str) -> str:
    if name not in AGGREGATION_CHOICES:
        raise ValueError(
            f"Unknown aggregation {name!r}; expected one of {AGGREGATION_CHOICES}"
        )
    return name
