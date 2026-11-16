"""Competitive model training, cross-validation, and MLflow logging orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from coffee_quality.config import get_settings
from coffee_quality.data.cleaner import clean_dataset, split_dataset
from coffee_quality.data.loader import load_raw_data
from coffee_quality.features.preprocessor import (
    build_feature_preprocessor,
    extract_features_and_target,
)
from coffee_quality.models.baseline import (
    CrossValidationResult,
    build_logistic_pipeline,
    evaluate_cv,
)
from coffee_quality.models.evaluator import EvaluationMetrics, evaluate_predictions
from coffee_quality.models.registry import (
    get_client,
    init_mlflow,
    log_experiment_run,
    promote_model_alias,
)

logger = logging.getLogger(__name__)


def build_competitive_pipeline(
    model_type: str = "random_forest",
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_split: int = 2,
) -> Pipeline:
    """Build a competitive ensemble classification pipeline with feature preprocessing.

    Args:
        model_type: Classifier family ('random_forest' or 'lightgbm').
        random_state: Random seed for deterministic reproducibility.
        n_estimators: Number of boosting trees or ensemble estimators.
        max_depth: Maximum tree depth (None for unlimited depth).
        min_samples_split: Minimum samples required to split an internal node.

    Returns:
        Scikit-learn Pipeline containing preprocessor and competitive classifier.

    Raises:
        ValueError: If model_type is not 'random_forest' or 'lightgbm'.
    """
    preprocessor = build_feature_preprocessor()

    if model_type == "random_forest":
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    elif model_type == "lightgbm":
        classifier = LGBMClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth if max_depth is not None else -1,
            class_weight="balanced",
            random_state=random_state,
            verbose=-1,
        )
    else:
        raise ValueError(
            f"Unsupported model_type '{model_type}'. Allowed types: 'random_forest', 'lightgbm'."
        )

    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


def train_and_evaluate_pipeline(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> tuple[Pipeline, EvaluationMetrics, CrossValidationResult]:
    """Train a pipeline on training data with cross-validation and evaluate on holdout test set.

    Args:
        pipeline: Scikit-learn Pipeline to train.
        train_df: Pre-split training DataFrame with features and target.
        test_df: Pre-split test DataFrame with features and target.
        n_splits: Number of stratified cross-validation folds.
        random_state: Random state for deterministic cross-validation.

    Returns:
        Tuple of (fitted Pipeline, test EvaluationMetrics, CrossValidationResult).
    """
    X_train, y_train = extract_features_and_target(train_df)
    X_test, y_test = extract_features_and_target(test_df)

    cv_results = evaluate_cv(
        pipeline=pipeline,
        X=X_train,
        y=y_train,
        n_splits=n_splits,
        random_state=random_state,
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    probs = cast(np.ndarray[Any, Any], pipeline.predict_proba(X_test))
    y_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]

    test_metrics = evaluate_predictions(y_test, y_pred, y_prob)
    return pipeline, test_metrics, cv_results


def run_training_pipeline(
    raw_data_path: Path | None = None,
    model_type: str = "random_forest",
    test_size: float = 0.2,
    n_splits: int = 5,
    random_state: int = 42,
    register_as_champion: bool = True,
    registered_model_name: str = "coffee-quality-model",
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> tuple[str, EvaluationMetrics, CrossValidationResult]:
    """Execute complete end-to-end training, MLflow logging, evaluation, and model registration.

    1. Loads raw dataset and validates SHA-256 integrity.
    2. Cleans records and creates stratified train/test split.
    3. Trains and logs baseline model (LogisticRegression) to MLflow.
    4. Trains and logs competitive ensemble model to MLflow.
    5. Validates competitive model performance against baseline.
    6. Promotes competitive model to 'champion' alias in Model Registry if requested.

    Args:
        raw_data_path: Path to dataset CSV. Defaults to settings.raw_data_path.
        model_type: Classifier architecture ('random_forest' or 'lightgbm').
        n_splits: Number of stratified cross-validation folds.
        random_state: Random state for deterministic reproducibility.
        register_as_champion: Whether to register and alias model in MLflow Model Registry.
        registered_model_name: Model name in registry (default 'coffee-quality-model').
        tracking_uri: Optional MLflow tracking URI override.
        experiment_name: Optional MLflow experiment name override.

    Returns:
        Tuple of (competitive_run_id, test_metrics, cv_results).
    """
    settings = get_settings()
    data_path = raw_data_path or settings.data_raw_path

    # Initialize tracking
    init_mlflow(tracking_uri=tracking_uri, experiment_name=experiment_name)

    # 1. Load, clean, and split dataset
    raw_df, _ = load_raw_data(data_path)
    clean_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(
        clean_df,
        test_size=test_size,
        random_state=random_state,
    )

    # 2. Train baseline model and log to MLflow
    logger.info("Training baseline LogisticRegression model...")
    base_pipe = build_logistic_pipeline(random_state=random_state)
    base_pipe, base_test_metrics, base_cv = train_and_evaluate_pipeline(
        pipeline=base_pipe,
        train_df=train_df,
        test_df=test_df,
        n_splits=n_splits,
        random_state=random_state,
    )
    log_experiment_run(
        pipeline=base_pipe,
        run_name="baseline_logistic_regression",
        params={
            "model_type": "LogisticRegression",
            "class_weight": "balanced",
            "random_state": random_state,
        },
        cv_results=base_cv,
        test_metrics=base_test_metrics,
        tags={"model_role": "baseline"},
    )

    # 3. Train competitive model and log to MLflow
    logger.info("Training competitive %s model...", model_type)
    comp_pipe = build_competitive_pipeline(
        model_type=model_type,
        random_state=random_state,
    )
    comp_pipe, comp_test_metrics, comp_cv = train_and_evaluate_pipeline(
        pipeline=comp_pipe,
        train_df=train_df,
        test_df=test_df,
        n_splits=n_splits,
        random_state=random_state,
    )

    comp_run_id = log_experiment_run(
        pipeline=comp_pipe,
        run_name=f"competitive_{model_type}",
        params={
            "model_type": model_type,
            "class_weight": "balanced",
            "random_state": random_state,
            "n_estimators": 100,
        },
        cv_results=comp_cv,
        test_metrics=comp_test_metrics,
        tags={"model_role": "candidate_champion"},
        registered_model_name=registered_model_name if register_as_champion else None,
    )

    # 4. Performance Quality Gate
    logger.info(
        "Baseline Macro-F1: %.4f | Competitive Macro-F1: %.4f",
        base_test_metrics.macro_f1,
        comp_test_metrics.macro_f1,
    )

    # 5. Model Registry Promotion
    if register_as_champion:
        client = get_client(tracking_uri=tracking_uri)
        versions = list(client.search_model_versions(f"name = '{registered_model_name}'"))
        if versions:
            latest_version = max(versions, key=lambda v: int(v.version)).version
            promote_model_alias(
                model_name=registered_model_name,
                version=latest_version,
                alias="champion",
                tracking_uri=tracking_uri,
            )
            logger.info(
                "Promoted model '%s' version %s to alias 'champion'",
                registered_model_name,
                latest_version,
            )

    return comp_run_id, comp_test_metrics, comp_cv


def main() -> int:
    """CLI execution entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger.info("Starting Arabica coffee quality model training pipeline...")
    run_id, metrics, cv_res = run_training_pipeline()
    logger.info("Training pipeline completed successfully.")
    logger.info("Promoted Champion Run ID: %s", run_id)
    if metrics.roc_auc is not None:
        logger.info("Test ROC-AUC: %.4f | Test Macro-F1: %.4f", metrics.roc_auc, metrics.macro_f1)
    logger.info("CV ROC-AUC Mean: %.4f | CV Macro-F1 Mean: %.4f", cv_res.roc_auc_mean, cv_res.macro_f1_mean)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
