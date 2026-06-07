from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DatasetSpec, FeatureSelectionConfig


@dataclass
class PreparedDataset:
    name: str
    frame: pd.DataFrame
    feature_columns: list[str]
    label_column: str
    date_column: str
    commit_key: str

    @property
    def X(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    @property
    def y(self) -> pd.Series:
        return self.frame[self.label_column].astype(int)


class DatasetRepository:
    """Loads multiple project datasets and applies a shared preprocessing policy."""

    def __init__(
        self,
        specs: list[DatasetSpec] | tuple[DatasetSpec, ...],
        feature_config: FeatureSelectionConfig | None = None,
    ) -> None:
        self.specs = list(specs)
        self.feature_config = feature_config or FeatureSelectionConfig()
        self.preprocessor = DatasetPreprocessor(self.feature_config)

    def load_all(self) -> list[PreparedDataset]:
        return [self.load(spec) for spec in self.specs]

    def load(self, spec: DatasetSpec) -> PreparedDataset:
        frame = pd.read_csv(Path(spec.feature_path))
        return self.preprocessor.prepare(frame, spec)


class DatasetPreprocessor:
    def __init__(self, config: FeatureSelectionConfig) -> None:
        self.config = config

    def prepare(self, df: pd.DataFrame, spec: DatasetSpec) -> PreparedDataset:
        required = [spec.commit_key, spec.date_column, spec.label_column]
        missing = [column for column in required if column not in df.columns]
        if missing:
            raise ValueError(
                f"{spec.name}: cleaned dataset is missing required columns {missing}."
            )

        frame = df.copy()
        frame[spec.date_column] = pd.to_datetime(frame[spec.date_column], errors="coerce", utc=True)
        if frame[[spec.date_column, spec.label_column]].isna().any().any():
            raise ValueError(f"{spec.name}: cleaned dataset contains missing date or label values.")
        frame[spec.label_column] = frame[spec.label_column].astype(int)
        label_values = set(frame[spec.label_column].unique())
        if not label_values.issubset({0, 1}):
            raise ValueError(f"{spec.name}: label column must contain only 0/1 values, got {sorted(label_values)}.")
        frame = frame.sort_values(spec.date_column).reset_index(drop=True)

        feature_columns = self._infer_feature_columns(frame, spec)
        if not feature_columns:
            raise ValueError(f"{spec.name}: no configured feature columns are usable.")

        if frame[feature_columns].isna().any().any():
            raise ValueError(f"{spec.name}: cleaned dataset contains missing feature values.")
        feature_values = frame[feature_columns].to_numpy(dtype=float)
        if not np.isfinite(feature_values).all():
            raise ValueError(f"{spec.name}: cleaned dataset contains infinite feature values.")
        return PreparedDataset(
            name=spec.name,
            frame=frame,
            feature_columns=feature_columns,
            label_column=spec.label_column,
            date_column=spec.date_column,
            commit_key=spec.commit_key,
        )

    def _infer_feature_columns(self, frame: pd.DataFrame, spec: DatasetSpec) -> list[str]:
        excluded = {
            spec.commit_key,
            spec.date_column,
            spec.label_column,
        }
        numeric_columns = set(frame.select_dtypes(include=["number", "bool"]).columns)
        if not self.config.include_columns:
            raise ValueError(f"{spec.name}: feature columns must be configured explicitly.")
        missing_includes = [column for column in self.config.include_columns if column not in frame.columns]
        if missing_includes:
            raise ValueError(f"{spec.name}: configured feature columns are missing: {missing_includes}")
        candidate_columns = list(self.config.include_columns)

        feature_columns: list[str] = []

        for column in candidate_columns:
            if column in excluded:
                continue
            if column not in numeric_columns:
                raise ValueError(f"{spec.name}: configured feature column '{column}' is not numeric or boolean.")
            feature_columns.append(column)

        return feature_columns
