from __future__ import annotations

from dataclasses import dataclass
import re

import pandas as pd


STAT_SUFFIXES = ("mean", "max", "min", "std")


@dataclass(frozen=True)
class FeatureConcept:
    name: str
    members: tuple[str, ...]
    grouped: bool


class FeatureGrouper:
    """Maps raw commit-level aggregate columns to concept-level metrics."""

    def __init__(self, feature_columns: list[str] | tuple[str, ...]) -> None:
        self.feature_columns = list(feature_columns)
        self.concepts = self._build_concepts()

    def _build_concepts(self) -> dict[str, FeatureConcept]:
        buckets: dict[str, list[str]] = {}
        for column in self.feature_columns:
            buckets.setdefault(self.normalize_concept_name(column), []).append(column)

        concepts: dict[str, FeatureConcept] = {}
        for concept_name, members in sorted(buckets.items()):
            concepts[concept_name] = FeatureConcept(
                name=concept_name,
                members=tuple(sorted(members)),
                grouped=len(members) > 1,
            )
        return concepts

    @staticmethod
    def normalize_concept_name(column: str) -> str:
        if column.startswith("current_"):
            stripped = column[len("current_") :]
            for suffix in STAT_SUFFIXES:
                marker = f"_{suffix}"
                if stripped.endswith(marker):
                    return stripped[: -len(marker)]
            return column

        for suffix in STAT_SUFFIXES:
            marker = f"_{suffix}"
            if column.endswith(marker):
                return column[: -len(marker)]

        return column

    def members_for_concepts(self, concepts: set[str] | list[str] | tuple[str, ...]) -> list[str]:
        selected: list[str] = []
        for concept_name in concepts:
            concept = self.concepts.get(concept_name)
            if concept is not None:
                selected.extend(concept.members)
        return [column for column in self.feature_columns if column in set(selected)]

    def concept_metadata(self) -> pd.DataFrame:
        rows = []
        for concept_name, concept in self.concepts.items():
            rows.append(
                {
                    "concept": concept_name,
                    "grouped": concept.grouped,
                    "member_count": len(concept.members),
                    "members": "|".join(concept.members),
                }
            )
        return pd.DataFrame(rows)

    def resolve_family_columns(
        self,
        concepts: tuple[str, ...] = (),
        columns: tuple[str, ...] = (),
        include_patterns: tuple[str, ...] = (),
        exclude_patterns: tuple[str, ...] = (),
    ) -> list[str]:
        selected = set(column for column in columns if column in self.feature_columns)
        selected.update(self.members_for_concepts(concepts))

        compiled_includes = [re.compile(pattern) for pattern in include_patterns]
        compiled_excludes = [re.compile(pattern) for pattern in exclude_patterns]

        if compiled_includes:
            for column in self.feature_columns:
                concept = self.normalize_concept_name(column)
                if any(pattern.search(column) or pattern.search(concept) for pattern in compiled_includes):
                    selected.add(column)

        if compiled_excludes:
            selected = {
                column
                for column in selected
                if not any(
                    pattern.search(column) or pattern.search(self.normalize_concept_name(column))
                    for pattern in compiled_excludes
                )
            }

        return [column for column in self.feature_columns if column in selected]

    def max_pool_scores(self, raw_scores: pd.Series) -> pd.Series:
        pooled = {}
        for concept_name, concept in self.concepts.items():
            values = raw_scores.reindex(concept.members).dropna()
            pooled[concept_name] = values.max() if not values.empty else pd.NA
        return pd.Series(pooled, dtype="float64")

    def max_pool_representatives(self, raw_scores: pd.Series) -> pd.DataFrame:
        rows = []
        for concept_name, concept in self.concepts.items():
            values = raw_scores.reindex(concept.members).dropna()
            if values.empty:
                representative = pd.NA
                score = pd.NA
            else:
                representative = str(values.idxmax())
                score = float(values.max())
            rows.append(
                {
                    "concept": concept_name,
                    "representative_feature": representative,
                    "representative_score": score,
                    "grouped": concept.grouped,
                    "members": "|".join(concept.members),
                }
            )
        return pd.DataFrame(rows)

    def sum_importances(self, raw_importances: pd.Series) -> pd.Series:
        grouped = {}
        for concept_name, concept in self.concepts.items():
            values = raw_importances.reindex(concept.members).fillna(0.0)
            grouped[concept_name] = float(values.sum())
        return pd.Series(grouped, dtype="float64")
