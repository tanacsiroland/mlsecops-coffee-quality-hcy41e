"""Unit tests for baseline models and cross-validation harness."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from coffee_quality.data.cleaner import clean_dataset, split_dataset
from coffee_quality.data.loader import load_raw_data
from coffee_quality.features.preprocessor import extract_features_and_target
from coffee_quality.models.baseline import (
    CrossValidationResult,
    build_dummy_pipeline,
    build_logistic_pipeline,
    evaluate_cv,
    train_and_evaluate_baseline,
)
from coffee_quality.models.evaluator import EvaluationMetrics


class NoProbaClassifier(BaseEstimator, ClassifierMixin):
    """Minimal estimator without predict_proba for testing fallback handling."""

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[no-untyped-def]
        self.classes_ = [0, 1]
        return self

    def predict(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        return [1] * len(X)


class SingleColProbaClassifier(BaseEstimator, ClassifierMixin):
    """Estimator whose predict_proba returns a single column."""

    def fit(self, X: pd.DataFrame, y: pd.Series):  # type: ignore[no-untyped-def]
        self.classes_ = [1]
        return self

    def predict(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        return [1] * len(X)

    def predict_proba(self, X: pd.DataFrame):  # type: ignore[no-untyped-def]
        return np.ones((len(X), 1))


@pytest.fixture
def sample_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned and split dataset for testing."""
    raw_path = Path("data/raw/dataset.csv")
    raw_df, _ = load_raw_data(raw_path)
    clean_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(clean_df, test_size=0.2, random_state=42)
    return train_df, test_df


def test_build_dummy_pipeline():
    """Verify dummy pipeline structure and naive predictions."""
    pipeline = build_dummy_pipeline()
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_build_logistic_pipeline():
    """Verify logistic pipeline parameter configuration."""
    pipeline = build_logistic_pipeline(random_state=123, max_iter=500, c=0.5)
    classifier = pipeline.named_steps["classifier"]
    assert classifier.random_state == 123
    assert classifier.max_iter == 500
    assert classifier.C == 0.5
    assert classifier.class_weight == "balanced"


def test_evaluate_cv_validation_errors():
    """Test parameter validation in evaluate_cv."""
    pipeline = build_dummy_pipeline()
    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 1, 1])

    with pytest.raises(ValueError, match="n_splits must be at least 2"):
        evaluate_cv(pipeline, X, y, n_splits=1)

    with pytest.raises(ValueError, match="Length mismatch"):
        evaluate_cv(pipeline, X, pd.Series([0, 1]))


def test_evaluate_cv_dummy_baseline(sample_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    """Verify that the dummy baseline has 0 commercial recall and low macro-F1 across folds."""
    train_df, _ = sample_dataset
    X, y = extract_features_and_target(train_df)

    dummy_pipeline = build_dummy_pipeline()
    cv_results = evaluate_cv(dummy_pipeline, X, y, n_splits=3, random_state=42)

    assert isinstance(cv_results, CrossValidationResult)
    assert cv_results.recall_commercial_mean == 0.0
    assert cv_results.macro_f1_mean < 0.50
    assert cv_results.accuracy_mean > 0.80  # Demonstrates accuracy trap

    d = cv_results.to_dict()
    assert "macro_f1_mean" in d
    assert "fold_metrics" not in d


def test_evaluate_cv_no_proba_classifier(sample_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    """Verify handling when an estimator lacks predict_proba."""
    train_df, _ = sample_dataset
    X, y = extract_features_and_target(train_df)

    from sklearn.pipeline import Pipeline
    from coffee_quality.features.preprocessor import build_feature_preprocessor

    pipeline = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", NoProbaClassifier()),
    ])

    cv_results = evaluate_cv(pipeline, X, y, n_splits=2, random_state=42)
    assert cv_results.roc_auc_mean == 0.0


def test_evaluate_cv_single_col_proba(sample_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    """Verify handling when an estimator returns single-column probabilities."""
    train_df, _ = sample_dataset
    X, y = extract_features_and_target(train_df)

    from sklearn.pipeline import Pipeline
    from coffee_quality.features.preprocessor import build_feature_preprocessor

    pipeline = Pipeline([
        ("preprocessor", build_feature_preprocessor()),
        ("classifier", SingleColProbaClassifier()),
    ])

    cv_results = evaluate_cv(pipeline, X, y, n_splits=2, random_state=42)
    assert isinstance(cv_results, CrossValidationResult)


def test_train_and_evaluate_baseline(sample_dataset: tuple[pd.DataFrame, pd.DataFrame]):
    """Verify end-to-end baseline training, cross-validation, and holdout evaluation."""
    train_df, test_df = sample_dataset

    pipeline, test_metrics, cv_results = train_and_evaluate_baseline(
        train_df,
        test_df,
        random_state=42,
        n_splits=3,
    )

    # Cross-validation performance checks
    assert cv_results.roc_auc_mean > 0.65
    assert cv_results.macro_f1_mean > 0.50
    assert cv_results.recall_commercial_mean > 0.30  # Balanced weighting learns commercial signals

    # Holdout test set checks
    assert isinstance(test_metrics, EvaluationMetrics)
    assert test_metrics.roc_auc is not None
    assert test_metrics.roc_auc > 0.65
    assert test_metrics.macro_f1 > 0.50
    assert test_metrics.true_negatives > 0
    assert test_metrics.true_positives > 0

    # Ensure model is fitted and can predict on unseen arbitrary test records
    X_test, _ = extract_features_and_target(test_df.iloc[:5])
    preds = pipeline.predict(X_test)
    assert len(preds) == 5
