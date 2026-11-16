"""Unit tests for competitive model training and orchestration pipeline."""

from pathlib import Path

import pandas as pd
import pytest
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from coffee_quality.data.cleaner import clean_dataset, split_dataset
from coffee_quality.data.loader import load_raw_data
from coffee_quality.models.evaluator import EvaluationMetrics
from coffee_quality.models.registry import get_client, load_registered_model
from coffee_quality.models.trainer import (
    build_competitive_pipeline,
    run_training_pipeline,
    train_and_evaluate_pipeline,
)


@pytest.fixture
def tracking_uri(tmp_path: Path) -> str:
    """Create an isolated SQLite tracking URI for testing."""
    db_file = tmp_path / "test_trainer.db"
    return f"sqlite:///{db_file}"


@pytest.fixture
def sample_splits() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cleaned dataset and generate reproducible train and test splits."""
    raw_path = Path("data/raw/dataset.csv")
    raw_df, _ = load_raw_data(raw_path)
    clean_df = clean_dataset(raw_df)
    train_df, test_df = split_dataset(clean_df, test_size=0.2, random_state=42)
    return train_df, test_df


def test_build_competitive_pipeline_random_forest():
    """Verify RandomForest pipeline assembly and parameter assignment."""
    pipeline = build_competitive_pipeline(
        model_type="random_forest",
        random_state=123,
        n_estimators=50,
        max_depth=10,
        min_samples_split=4,
    )
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, RandomForestClassifier)
    assert classifier.random_state == 123
    assert classifier.n_estimators == 50
    assert classifier.max_depth == 10
    assert classifier.min_samples_split == 4
    assert classifier.class_weight == "balanced"


def test_build_competitive_pipeline_lightgbm():
    """Verify LightGBM pipeline assembly and parameter assignment."""
    pipeline = build_competitive_pipeline(
        model_type="lightgbm",
        random_state=77,
        n_estimators=40,
        max_depth=5,
    )
    assert isinstance(pipeline, Pipeline)
    classifier = pipeline.named_steps["classifier"]
    assert isinstance(classifier, LGBMClassifier)
    assert classifier.random_state == 77
    assert classifier.n_estimators == 40
    assert classifier.max_depth == 5
    assert classifier.class_weight == "balanced"


def test_build_competitive_pipeline_unsupported_raises():
    """Verify that unsupported model types raise a descriptive ValueError."""
    with pytest.raises(ValueError, match="Unsupported model_type 'unsupported'"):
        build_competitive_pipeline(model_type="unsupported")


def test_train_and_evaluate_pipeline(sample_splits: tuple[pd.DataFrame, pd.DataFrame]):
    """Verify train_and_evaluate_pipeline fits estimator and returns valid metrics."""
    train_df, test_df = sample_splits
    pipeline = build_competitive_pipeline(
        model_type="random_forest",
        n_estimators=10,
        random_state=42,
    )

    fitted_pipe, test_metrics, cv_results = train_and_evaluate_pipeline(
        pipeline=pipeline,
        train_df=train_df,
        test_df=test_df,
        n_splits=2,
        random_state=42,
    )

    assert isinstance(fitted_pipe, Pipeline)
    assert isinstance(test_metrics, EvaluationMetrics)
    assert test_metrics.roc_auc is not None
    assert test_metrics.roc_auc > 0.60
    assert test_metrics.macro_f1 > 0.50
    assert cv_results.roc_auc_mean > 0.60


def test_run_training_pipeline_end_to_end(tracking_uri: str):
    """Verify complete end-to-end training, MLflow tracking, and model promotion."""
    model_name = "coffee-quality-champion-test"
    run_id, test_metrics, cv_res = run_training_pipeline(
        raw_data_path=Path("data/raw/dataset.csv"),
        model_type="random_forest",
        test_size=0.2,
        n_splits=2,
        random_state=42,
        register_as_champion=True,
        registered_model_name=model_name,
        tracking_uri=tracking_uri,
        experiment_name="test_trainer_exp",
    )

    assert isinstance(run_id, str)
    assert test_metrics.roc_auc is not None
    assert test_metrics.roc_auc > 0.70
    assert test_metrics.macro_f1 > 0.55
    assert cv_res.roc_auc_mean > 0.70

    # Verify both baseline and competitive runs exist in MLflow
    client = get_client(tracking_uri=tracking_uri)
    exp = client.get_experiment_by_name("test_trainer_exp")
    assert exp is not None
    runs = client.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 2  # Baseline and competitive

    # Verify champion alias promotion and model loading
    loaded_champion = load_registered_model(
        model_name=model_name,
        alias="champion",
        tracking_uri=tracking_uri,
    )
    assert isinstance(loaded_champion, Pipeline)


def test_run_training_pipeline_without_registration(tracking_uri: str):
    """Verify pipeline runs smoothly when register_as_champion is False."""
    run_id, test_metrics, cv_res = run_training_pipeline(
        raw_data_path=Path("data/raw/dataset.csv"),
        model_type="random_forest",
        test_size=0.2,
        n_splits=2,
        random_state=42,
        register_as_champion=False,
        tracking_uri=tracking_uri,
        experiment_name="test_no_reg_exp",
    )

    assert isinstance(run_id, str)
    assert test_metrics.accuracy > 0.70


def test_main_cli_execution(tracking_uri: str, monkeypatch: pytest.MonkeyPatch):
    """Test CLI main entry point execution."""
    from coffee_quality.models import trainer

    monkeypatch.setenv("COFFEE_MLFLOW_TRACKING_URI", tracking_uri)
    exit_code = trainer.main()
    assert exit_code == 0
