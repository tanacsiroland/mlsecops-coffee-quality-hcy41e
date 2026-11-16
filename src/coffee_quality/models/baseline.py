"""Baseline classification models and cross-validation evaluation harness."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from coffee_quality.features.preprocessor import (
    build_feature_preprocessor,
    extract_features_and_target,
)
from coffee_quality.models.evaluator import EvaluationMetrics, evaluate_predictions


@dataclass(frozen=True)
class CrossValidationResult:
    """Aggregated cross-validation metrics across all evaluation folds.

    Attributes:
        roc_auc_mean: Mean ROC-AUC score across folds.
        roc_auc_std: Standard deviation of ROC-AUC across folds.
        macro_f1_mean: Mean Macro-F1 score across folds.
        macro_f1_std: Standard deviation of Macro-F1 across folds.
        accuracy_mean: Mean accuracy score across folds.
        accuracy_std: Standard deviation of accuracy across folds.
        recall_commercial_mean: Mean recall for class 0 (Commercial).
        recall_commercial_std: Standard deviation of recall for class 0.
        precision_commercial_mean: Mean precision for class 0 (Commercial).
        precision_commercial_std: Standard deviation of precision for class 0.
        fold_metrics: List of individual EvaluationMetrics for each fold.
    """

    roc_auc_mean: float
    roc_auc_std: float
    macro_f1_mean: float
    macro_f1_std: float
    accuracy_mean: float
    accuracy_std: float
    recall_commercial_mean: float
    recall_commercial_std: float
    precision_commercial_mean: float
    precision_commercial_std: float
    fold_metrics: list[EvaluationMetrics]

    def to_dict(self) -> dict[str, float]:
        """Return a dictionary of mean and std metrics for experiment tracking."""
        data = asdict(self)
        del data["fold_metrics"]
        return cast(dict[str, float], data)


def build_dummy_pipeline() -> Pipeline:
    """Build a naive baseline pipeline predicting the most frequent class.

    Returns:
        Scikit-learn Pipeline with feature preprocessing and DummyClassifier.
    """
    preprocessor = build_feature_preprocessor()
    classifier = DummyClassifier(strategy="most_frequent")
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def build_logistic_pipeline(
    random_state: int = 42,
    max_iter: int = 1000,
    c: float = 1.0,
) -> Pipeline:
    """Build a balanced linear classification pipeline.

    Args:
        random_state: Random state for deterministic solver reproducibility.
        max_iter: Maximum iterations for solver convergence.
        c: Inverse regularization strength parameter.

    Returns:
        Scikit-learn Pipeline with feature preprocessing and LogisticRegression.
    """
    preprocessor = build_feature_preprocessor()
    classifier = LogisticRegression(
        class_weight="balanced",
        random_state=random_state,
        max_iter=max_iter,
        C=c,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def evaluate_cv(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
) -> CrossValidationResult:
    """Run stratified cross-validation and compute aggregated performance metrics.

    Fits the feature preprocessor and classifier strictly inside each fold,
    preventing any cross-fold information leakage.

    Args:
        pipeline: Scikit-learn Pipeline to fit and evaluate.
        X: Feature DataFrame.
        y: Binary target Series (0 or 1).
        n_splits: Number of stratified cross-validation folds (default 5).
        random_state: Random state for reproducible fold generation.

    Returns:
        CrossValidationResult containing fold metrics and aggregated means/stds.

    Raises:
        ValueError: If n_splits < 2 or X and y lengths mismatch.
    """
    if n_splits < 2:
        raise ValueError(f"n_splits must be at least 2, got {n_splits}.")
    if len(X) != len(y):
        raise ValueError(f"Length mismatch: X has {len(X)} rows, y has {len(y)} labels.")

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_metrics: list[EvaluationMetrics] = []

    y_clean = y.reset_index(drop=True)
    X_clean = X.reset_index(drop=True)

    for train_idx, val_idx in skf.split(X_clean, y_clean):
        X_train_fold = cast(pd.DataFrame, X_clean.iloc[train_idx])
        y_train_fold = cast(pd.Series, y_clean.iloc[train_idx])
        X_val_fold = cast(pd.DataFrame, X_clean.iloc[val_idx])
        y_val_fold = cast(pd.Series, y_clean.iloc[val_idx])

        # Fit strictly on train fold
        pipeline.fit(X_train_fold, y_train_fold)

        y_pred = pipeline.predict(X_val_fold)
        y_prob: np.ndarray[Any, Any] | None = None
        if hasattr(pipeline, "predict_proba"):
            probs = cast(np.ndarray[Any, Any], pipeline.predict_proba(X_val_fold))
            if probs.shape[1] > 1:
                y_prob = probs[:, 1]
            else:
                # In single-class cases (e.g. DummyClassifier), probabilities only have 1 column
                y_prob = probs[:, 0]

        metrics = evaluate_predictions(y_val_fold, y_pred, y_prob)
        fold_metrics.append(metrics)

    roc_aucs = [m.roc_auc for m in fold_metrics if m.roc_auc is not None]
    roc_auc_mean = float(np.mean(roc_aucs)) if roc_aucs else 0.0
    roc_auc_std = float(np.std(roc_aucs)) if roc_aucs else 0.0

    macro_f1s = [m.macro_f1 for m in fold_metrics]
    accuracies = [m.accuracy for m in fold_metrics]
    recalls_comm = [m.recall_commercial for m in fold_metrics]
    precisions_comm = [m.precision_commercial for m in fold_metrics]

    return CrossValidationResult(
        roc_auc_mean=roc_auc_mean,
        roc_auc_std=roc_auc_std,
        macro_f1_mean=float(np.mean(macro_f1s)),
        macro_f1_std=float(np.std(macro_f1s)),
        accuracy_mean=float(np.mean(accuracies)),
        accuracy_std=float(np.std(accuracies)),
        recall_commercial_mean=float(np.mean(recalls_comm)),
        recall_commercial_std=float(np.std(recalls_comm)),
        precision_commercial_mean=float(np.mean(precisions_comm)),
        precision_commercial_std=float(np.std(precisions_comm)),
        fold_metrics=fold_metrics,
    )


def train_and_evaluate_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
    n_splits: int = 5,
) -> tuple[Pipeline, EvaluationMetrics, CrossValidationResult]:
    """Train the logistic regression baseline and evaluate on cross-validation and holdout test set.

    Args:
        train_df: Pre-split training DataFrame with features and target.
        test_df: Pre-split test DataFrame with features and target.
        random_state: Random state for deterministic reproducibility.
        n_splits: Number of cross-validation folds.

    Returns:
        Tuple of (fitted Pipeline, test EvaluationMetrics, CrossValidationResult).
    """
    X_train, y_train = extract_features_and_target(train_df)
    X_test, y_test = extract_features_and_target(test_df)

    pipeline = build_logistic_pipeline(random_state=random_state)

    cv_results = evaluate_cv(
        pipeline=pipeline,
        X=X_train,
        y=y_train,
        n_splits=n_splits,
        random_state=random_state,
    )

    # Fit final baseline on full training set
    pipeline.fit(X_train, y_train)

    y_test_pred = pipeline.predict(X_test)
    probs = cast(np.ndarray[Any, Any], pipeline.predict_proba(X_test))
    y_test_prob = probs[:, 1]

    test_metrics = evaluate_predictions(y_test, y_test_pred, y_test_prob)

    return pipeline, test_metrics, cv_results
