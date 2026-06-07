from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """One cleaned project-level commit dataset."""

    name: str
    feature_path: Path
    label_column: str
    commit_key: str = "commit_id"
    date_column: str = "commit_author_date"


@dataclass(frozen=True)
class FeatureSelectionConfig:
    """Explicit feature columns used by Stage 1/2/3."""

    include_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class NPSKConfig:
    alpha: float = 0.05
    rscript_path: str = "Rscript"
    r_version: str = "np"


@dataclass(frozen=True)
class Stage1Config:
    correlation_methods: tuple[str, ...] = ("spearman", "mutual_info", "kendall")
    absolute_correlation: bool = True
    random_state: int = 20260505
    npsk: NPSKConfig = field(default_factory=NPSKConfig)


@dataclass(frozen=True)
class TimeWindowConfig:
    initial_train_ratio: float = 0.50
    test_ratio: float = 0.10
    step_ratio: float = 0.10
    max_windows: int = 5
    expanding_train: bool = True
    min_train_size: int = 50
    min_test_size: int = 20


@dataclass(frozen=True)
class RandomForestConfig:
    tune_hyperparameters: bool = True
    n_estimators_grid: tuple[int, ...] = (100, 300, 500)
    max_features_grid: tuple[str | float | None, ...] = ("sqrt", 0.3, 0.5, None)
    min_samples_leaf_grid: tuple[int, ...] = (1, 3, 5)
    cv_splits: int = 3
    criterion: str = "gini"
    class_weight: str = "balanced_subsample"
    n_jobs: int = -1
    scoring: str = "gmean"


@dataclass(frozen=True)
class Stage2Config:
    time_windows: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    bootstrap_iterations: int = 20
    random_state: int = 20260505
    npsk: NPSKConfig = field(default_factory=NPSKConfig)


@dataclass(frozen=True)
class FeatureFamily:
    """Feature set definition for Stage 3.

    Use exact concept names, exact raw columns, or regex patterns. Patterns are matched
    against both raw feature columns and normalized concept names.
    """

    name: str
    concepts: tuple[str, ...] = ()
    columns: tuple[str, ...] = ()
    include_patterns: tuple[str, ...] = ()
    exclude_patterns: tuple[str, ...] = ()


@dataclass(frozen=True)
class Stage3Config:
    time_windows: TimeWindowConfig = field(default_factory=TimeWindowConfig)
    random_forest: RandomForestConfig = field(default_factory=RandomForestConfig)
    repeats: int = 5
    topk_percentages: tuple[int, ...] = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
    random_state: int = 20260505

