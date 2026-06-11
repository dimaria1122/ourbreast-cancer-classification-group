from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelResult:
    """Container for metrics and predictions used by tests and the notebook."""

    model: ClassifierMixin
    metrics: Dict[str, float]
    confusion_matrix: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray


def load_raw_dataset(path: str | Path) -> pd.DataFrame:
    """Load the original Kaggle CSV without changing its rows."""

    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Clean the Kaggle CSV and return feature matrix, target, and names."""

    cleaned = df.copy()
    empty_columns = [col for col in cleaned.columns if str(col).strip() == ""]
    unnamed_columns = [col for col in cleaned.columns if str(col).startswith("Unnamed")]
    drop_columns = ["id", *empty_columns, *unnamed_columns]
    cleaned = cleaned.drop(columns=[col for col in drop_columns if col in cleaned.columns])

    if "diagnosis" not in cleaned.columns:
        raise ValueError("Expected a 'diagnosis' column in the dataset.")

    y = cleaned["diagnosis"].map({"B": 0, "M": 1})
    if y.isna().any():
        bad_values = sorted(cleaned.loc[y.isna(), "diagnosis"].dropna().unique())
        raise ValueError(f"Unexpected diagnosis labels: {bad_values}")

    X = cleaned.drop(columns=["diagnosis"])
    X = X.apply(pd.to_numeric)
    feature_names = list(X.columns)
    return X, y.astype(int), feature_names


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load and clean the local Kaggle dataset."""

    return clean_dataset(load_raw_dataset(path))


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    """Create a deterministic stratified train/test split."""

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def build_models() -> Dict[str, ClassifierMixin]:
    """Return the four classifiers compared in the notebook."""

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        solver="liblinear",
                        max_iter=1000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "SVM": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
        "KNN": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
    }


def _score_positive_class(model: ClassifierMixin, X_test: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X_test)
        return np.asarray(scores, dtype=float)
    return model.predict(X_test)


def evaluate_models(
    models: Mapping[str, ClassifierMixin],
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> Dict[str, ModelResult]:
    """Fit all models and collect the metrics needed for comparison."""

    results: Dict[str, ModelResult] = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = _score_positive_class(model, X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_score),
        }
        results[name] = ModelResult(
            model=model,
            metrics=metrics,
            confusion_matrix=confusion_matrix(y_test, y_pred),
            y_pred=np.asarray(y_pred),
            y_score=np.asarray(y_score),
        )
    return results


def metrics_dataframe(results: Mapping[str, ModelResult]) -> pd.DataFrame:
    """Convert evaluation results into a sorted table."""

    rows = []
    for name, result in results.items():
        row = {"model": name}
        row.update(result.metrics)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True)
