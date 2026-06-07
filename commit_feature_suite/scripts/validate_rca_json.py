#!/usr/bin/env python3
"""Validate rust-code-analysis raw JSON field presence for key metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGET_FIELDS = [
    "halstead_n1",
    "halstead_n2",
    "halstead_N1",
    "halstead_N2",
    "halstead_volume",
    "halstead_difficulty",
    "halstead_effort",
    "halstead_bugs",
    "halstead_time",
    "mi",
    "nom",
]

ALIASES = {
    "halstead_n1": {"n1", "n_1", "distinct_operators"},
    "halstead_n2": {"n2", "n_2", "distinct_operands"},
    "halstead_N1": {"N1", "big_n1", "total_operators"},
    "halstead_N2": {"N2", "big_n2", "total_operands"},
    "halstead_volume": {"volume"},
    "halstead_difficulty": {"difficulty"},
    "halstead_effort": {"effort"},
    "halstead_bugs": {"bugs", "delivered_bugs"},
    "halstead_time": {"time", "time_required"},
    "mi": {"mi", "maintainability_index"},
    "nom": {"nom", "functions", "methods_count"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RCA JSON metric field hit rates.")
    parser.add_argument("--json_dir", required=True, help="Directory containing dumped RCA JSON files.")
    return parser.parse_args()


def find_by_alias(obj: Any, aliases: set[str]) -> bool:
    aliases_norm = {item.lower() for item in aliases}
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_norm = str(key).lower()
            if key_norm in aliases_norm or any(a in key_norm for a in aliases_norm):
                return True
            if find_by_alias(value, aliases_norm):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if find_by_alias(item, aliases_norm):
                return True
    return False


def main() -> int:
    args = parse_args()
    json_dir = Path(args.json_dir).expanduser().resolve()
    files = sorted(json_dir.glob("*.json"))
    if not files:
        print(f"No JSON files found in: {json_dir}")
        return 1

    hit_counts = {field: 0 for field in TARGET_FIELDS}
    total = 0
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        total += 1
        for field in TARGET_FIELDS:
            if find_by_alias(payload, ALIASES[field]):
                hit_counts[field] += 1

    print(f"Validated files: {total}")
    for field in TARGET_FIELDS:
        hit = hit_counts[field]
        rate = (hit / total * 100.0) if total else 0.0
        print(f"{field:24s} hit={hit:6d}  miss={total - hit:6d}  rate={rate:6.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

