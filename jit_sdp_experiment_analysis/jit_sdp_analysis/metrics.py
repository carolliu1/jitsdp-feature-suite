from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)


def gmean_score(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall1 = tp / (tp + fn) if (tp + fn) else 0.0
    recall0 = tn / (tn + fp) if (tn + fp) else 0.0
    return float(np.sqrt(recall1 * recall0))


GMEAN_SCORER = make_scorer(gmean_score)


def classification_metrics(y_true: pd.Series | np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray | None) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall1 = tp / (tp + fn) if (tp + fn) else 0.0
    recall0 = tn / (tn + fp) if (tn + fp) else 0.0
    pf_rate = fp / (fp + tn) if (fp + tn) else 0.0

    metrics = {
        "gmean": gmean_score(y_true, y_pred),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(set(y_pred)) > 1 else 0.0,
        "pd": float(recall1),
        "pf": float(pf_rate),
        "recall0": float(recall0),
        "recall1": float(recall1),
    }
    if y_prob is not None and len(set(y_true)) > 1:
        metrics["auc"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["auc"] = np.nan
    return metrics


class UnivariateScorer:
    def __init__(self, method: str, absolute_correlation: bool = True, random_state: int = 20260505) -> None:
        self.method = method
        self.absolute_correlation = absolute_correlation
        self.random_state = random_state

    def score(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        if self.method == "spearman":
            return self._rank_correlation(X, y, kind="spearman")
        if self.method == "kendall":
            return self._rank_correlation(X, y, kind="kendall")
        if self.method == "mutual_info":
            return self._mutual_information(X, y)
        raise ValueError(f"Unsupported univariate method: {self.method}")

    def _rank_correlation(self, X: pd.DataFrame, y: pd.Series, kind: str) -> pd.Series:
        scores = {}
        for column in X.columns:
            series = X[column]
            valid = ~(series.isna() | y.isna())
            if valid.sum() < 3 or series[valid].nunique() < 2 or y[valid].nunique() < 2:
                scores[column] = 0.0
                continue
            if kind == "spearman":
                value = spearmanr(series[valid], y[valid], nan_policy="omit").correlation
            else:
                value = kendalltau(series[valid], y[valid], nan_policy="omit").correlation
            if pd.isna(value):
                value = 0.0
            scores[column] = abs(float(value)) if self.absolute_correlation else float(value)
        return pd.Series(scores, dtype="float64")

    def _mutual_information(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        usable = [column for column in X.columns if X[column].dropna().nunique() >= 2]
        scores = pd.Series(0.0, index=X.columns, dtype="float64")
        if not usable or y.nunique() < 2:
            return scores

        values = mutual_info_classif(X[usable], y.astype(int), random_state=self.random_state)
        scores.loc[usable] = values
        return scores
