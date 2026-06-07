from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils import resample

from .config import RandomForestConfig, TimeWindowConfig
from .metrics import GMEAN_SCORER, classification_metrics


MAX_SKLEARN_RANDOM_STATE = 2**32 - 1


def normalize_random_state(random_state: int) -> int:
    return int(random_state % MAX_SKLEARN_RANDOM_STATE)


@dataclass(frozen=True)
class TimeWindow:
    window_id: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def to_record(self, dataset_name: str, frame: pd.DataFrame, date_column: str) -> dict[str, object]:
        train_frame = frame.iloc[self.train_start : self.train_end]
        test_frame = frame.iloc[self.test_start : self.test_end]
        return {
            "dataset": dataset_name,
            "window_id": self.window_id,
            "train_start_index": self.train_start,
            "train_end_index": self.train_end - 1,
            "test_start_index": self.test_start,
            "test_end_index": self.test_end - 1,
            "train_size": int(train_frame.shape[0]),
            "test_size": int(test_frame.shape[0]),
            "train_start_date": train_frame[date_column].iloc[0] if not train_frame.empty else pd.NaT,
            "train_end_date": train_frame[date_column].iloc[-1] if not train_frame.empty else pd.NaT,
            "test_start_date": test_frame[date_column].iloc[0] if not test_frame.empty else pd.NaT,
            "test_end_date": test_frame[date_column].iloc[-1] if not test_frame.empty else pd.NaT,
            "is_consecutive": self.test_start == self.train_end,
        }


class TimeForwardSplitter:
    def __init__(self, config: TimeWindowConfig) -> None:
        self.config = config

    def split(self, frame: pd.DataFrame) -> list[TimeWindow]:
        n_rows = len(frame)
        train_size = max(self.config.min_train_size, int(n_rows * self.config.initial_train_ratio))
        test_size = max(self.config.min_test_size, int(n_rows * self.config.test_ratio))
        step_size = max(1, int(n_rows * self.config.step_ratio))

        windows: list[TimeWindow] = []
        train_end = train_size
        window_id = 1

        while train_end + test_size <= n_rows and len(windows) < self.config.max_windows:
            if self.config.expanding_train:
                train_start = 0
            else:
                train_start = max(0, train_end - train_size)
            test_start = train_end
            test_end = train_end + test_size
            windows.append(
                TimeWindow(
                    window_id=window_id,
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                )
            )
            train_end += step_size
            window_id += 1

        return windows


class RandomForestService:
    def __init__(self, config: RandomForestConfig) -> None:
        self.config = config

    def tune(self, X_train: pd.DataFrame, y_train: pd.Series, random_state: int) -> dict[str, object]:
        default_params = {
            "n_estimators": self.config.n_estimators_grid[0],
            "max_features": self.config.max_features_grid[0],
            "min_samples_leaf": self.config.min_samples_leaf_grid[0],
        }
        if not self.config.tune_hyperparameters or y_train.nunique() < 2:
            return default_params

        n_splits = min(self.config.cv_splits, max(2, len(X_train) // 50))
        if n_splits < 2:
            return default_params

        pipeline = self._pipeline(default_params, random_state)
        param_grid = {
            "rf__n_estimators": list(self.config.n_estimators_grid),
            "rf__max_features": list(self.config.max_features_grid),
            "rf__min_samples_leaf": list(self.config.min_samples_leaf_grid),
        }
        search = GridSearchCV(
            pipeline,
            param_grid=param_grid,
            scoring=GMEAN_SCORER,
            cv=TimeSeriesSplit(n_splits=n_splits),
            n_jobs=self.config.n_jobs,
            error_score=np.nan,
            refit=True,
        )
        try:
            search.fit(X_train, y_train)
        except ValueError:
            return default_params

        best = search.best_params_
        return {
            "n_estimators": best["rf__n_estimators"],
            "max_features": best["rf__max_features"],
            "min_samples_leaf": best["rf__min_samples_leaf"],
        }

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, params: dict[str, object], random_state: int) -> Pipeline:
        model = self._pipeline(params, random_state)
        model.fit(X_train, y_train)
        return model

    def bootstrap_sample(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        random_state: int,
    ) -> tuple[pd.DataFrame, pd.Series]:
        stratify = y_train if y_train.nunique() == 2 and y_train.value_counts().min() > 1 else None
        X_resampled, y_resampled = resample(
            X_train,
            y_train,
            replace=True,
            stratify=stratify,
            random_state=normalize_random_state(random_state),
        )
        return X_resampled, y_resampled

    def evaluate(self, model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        y_pred = model.predict(X_test)
        y_prob = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X_test)
            if probabilities.shape[1] > 1:
                y_prob = probabilities[:, 1]
        return classification_metrics(y_test, y_pred, y_prob)

    def feature_importances(self, model: Pipeline, feature_columns: list[str]) -> pd.Series:
        rf = model.named_steps["rf"]
        # RandomForestClassifier.feature_importances_ is impurity-based MDI.
        return pd.Series(rf.feature_importances_, index=feature_columns, dtype="float64")

    def _pipeline(self, params: dict[str, object], random_state: int) -> Pipeline:
        rf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            criterion=self.config.criterion,
            max_features=params["max_features"],
            min_samples_leaf=int(params["min_samples_leaf"]),
            class_weight=self.config.class_weight,
            n_jobs=self.config.n_jobs,
            random_state=normalize_random_state(random_state),
        )
        return Pipeline(
            steps=[
                ("rf", rf),
            ]
        )
