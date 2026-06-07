"""Aggregation helpers for commit-level feature statistics."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def append_if_number(target: List[float], value: Any) -> None:
    if isinstance(value, (int, float)):
        target.append(float(value))


def stats_prefix(values: List[int] | List[float], prefix: str) -> Dict[str, float | None]:
    """Return mean/max/min/std for one value series."""
    if not values:
        return {
            f"{prefix}_mean": None,
            f"{prefix}_max": None,
            f"{prefix}_min": None,
            f"{prefix}_std": None,
        }
    series = pd.Series(values, dtype="float64")
    return {
        f"{prefix}_mean": float(series.mean()),
        f"{prefix}_max": float(series.max()),
        f"{prefix}_min": float(series.min()),
        f"{prefix}_std": float(series.std(ddof=0)),
    }

