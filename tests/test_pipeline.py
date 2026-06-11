from pathlib import Path

import pytest

from src.breast_cancer_pipeline import (
    build_models,
    evaluate_models,
    load_dataset,
    split_dataset,
)


DATA_PATH = Path(__file__).resolve().parents[1] / "data.csv"


def test_load_dataset_cleans_kaggle_csv():
    X, y, feature_names = load_dataset(DATA_PATH)

    assert X.shape == (569, 30)
    assert y.shape == (569,)
    assert set(y.unique()) == {0, 1}
    assert len(feature_names) == 30
    assert "id" not in feature_names
    assert all(not str(name).startswith("Unnamed") for name in feature_names)
    assert all(str(name).strip() for name in feature_names)


def test_build_models_contains_required_four_classifiers():
    models = build_models()

    assert list(models) == [
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "KNN",
    ]


@pytest.mark.filterwarnings("ignore:.*X does not have valid feature names.*")
def test_evaluate_models_returns_complete_metrics():
    X, y, _ = load_dataset(DATA_PATH)
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    results = evaluate_models(build_models(), X_train, X_test, y_train, y_test)

    assert set(results) == {
        "Logistic Regression",
        "SVM",
        "Random Forest",
        "KNN",
    }

    metric_names = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    for result in results.values():
        assert metric_names.issubset(result.metrics)
        for metric_name in metric_names:
            assert 0.0 <= result.metrics[metric_name] <= 1.0
        assert result.confusion_matrix.shape == (2, 2)
        assert result.y_pred.shape == y_test.shape
        assert result.y_score.shape == y_test.shape
        assert result.metrics["accuracy"] >= 0.90
