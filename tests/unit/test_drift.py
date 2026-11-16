"""Unit tests for the EvidentlyAI data and target drift detection engine."""

from pathlib import Path

import pandas as pd
import pytest

from coffee_quality.config import Settings
from coffee_quality.monitoring.drift import (
    DriftDetector,
    DriftReportSummary,
    FeatureDriftMetric,
)


@pytest.fixture
def sample_reference_df() -> pd.DataFrame:
    """Provide a diverse synthetic reference DataFrame."""
    n_samples = 60
    countries = ["Ethiopia", "Colombia", "Brazil", "Guatemala", "Kenya"]
    varieties = ["Typica", "Bourbon", "Caturra", "SL28", "Geisha"]
    methods = ["Washed", "Natural", "Washed", "Semi-Washed", "Natural"]
    colors = ["Green", "Bluish-Green", "Green", "Green", "Bluish-Green"]

    return pd.DataFrame({
        "Country.of.Origin": [countries[i % len(countries)] for i in range(n_samples)],
        "Variety": [varieties[i % len(varieties)] for i in range(n_samples)],
        "Processing.Method": [methods[i % len(methods)] for i in range(n_samples)],
        "Color": [colors[i % len(colors)] for i in range(n_samples)],
        "altitude_mean_meters": [1100.0 + (i * 20.0) for i in range(n_samples)],
        "Moisture": [0.09 + (i % 5) * 0.01 for i in range(n_samples)],
        "Category.One.Defects": [i % 3 for i in range(n_samples)],
        "Category.Two.Defects": [i % 5 for i in range(n_samples)],
        "Quakers": [0 if i % 4 != 0 else 1 for i in range(n_samples)],
        "Grading.Date": [f"2020-0{(i % 8) + 1}-15" for i in range(n_samples)],
        "Is_Specialty": [1 if i % 2 == 0 else 0 for i in range(n_samples)],
    })


def test_feature_drift_metric_properties() -> None:
    """Verify FeatureDriftMetric attributes."""
    metric = FeatureDriftMetric(
        column_name="altitude_mean_meters",
        method="ks_test",
        threshold=0.05,
        score=0.01,
        drift_detected=True,
    )
    assert metric.column_name == "altitude_mean_meters"
    assert metric.drift_detected is True
    assert metric.score == 0.01


def test_drift_detector_initialization(sample_reference_df: pd.DataFrame) -> None:
    """Verify initialization and custom thresholds."""
    detector = DriftDetector(sample_reference_df, drift_share_threshold=0.4, min_current_samples=8)
    assert detector.drift_share_threshold == 0.4
    assert detector.min_current_samples == 8
    assert len(detector.reference_data) == 60


def test_drift_detector_from_settings(tmp_path: Path, sample_reference_df: pd.DataFrame) -> None:
    """Verify factory instantiation loading reference dataset from settings."""
    raw_csv = tmp_path / "reference_raw.csv"
    sample_reference_df.to_csv(raw_csv, index=False)

    settings = Settings(data_raw_path=raw_csv)
    detector = DriftDetector.from_settings(settings=settings)
    assert len(detector.reference_data) == 60


def test_drift_detector_insufficient_samples(sample_reference_df: pd.DataFrame) -> None:
    """Verify guard when fewer than min_current_samples are provided."""
    detector = DriftDetector(sample_reference_df, min_current_samples=5)
    tiny_df = sample_reference_df.iloc[:3].copy()

    summary, html = detector.calculate_drift(tiny_df)

    assert isinstance(summary, DriftReportSummary)
    assert summary.status == "insufficient_samples"
    assert summary.dataset_drift_detected is False
    assert summary.current_sample_count == 3
    assert "Insufficient Samples" in html

    summary_dict = summary.to_dict()
    assert summary_dict["status"] == "insufficient_samples"


def test_drift_detector_no_drift_on_similar_distribution(sample_reference_df: pd.DataFrame) -> None:
    """Verify that similar distributions do not trigger dataset drift."""
    detector = DriftDetector(sample_reference_df, drift_share_threshold=0.5)

    # Use a 30-sample subset with matching distribution
    cur_df = sample_reference_df.iloc[:30].copy()

    summary, html = detector.calculate_drift(cur_df)

    assert summary.status == "success"
    assert summary.dataset_drift_detected is False
    assert summary.reference_sample_count == 60
    assert summary.current_sample_count == 30
    assert summary.total_features_count > 0
    assert len(html) > 500


def test_drift_detector_detects_severe_distribution_drift(sample_reference_df: pd.DataFrame) -> None:
    """Verify that severe multi-feature perturbations trigger dataset drift."""
    detector = DriftDetector(sample_reference_df, drift_share_threshold=0.4)

    # Construct heavily shifted current dataset
    drifted_df = sample_reference_df.copy()
    drifted_df["altitude_mean_meters"] = 3950.0  # extreme elevation shift
    drifted_df["Moisture"] = 0.01  # extreme dehydration
    drifted_df["Country.of.Origin"] = "SyntheticState"
    drifted_df["Category.One.Defects"] = 25
    drifted_df["Is_Specialty"] = 0  # complete target shift

    summary, html = detector.calculate_drift(drifted_df)

    assert summary.status == "success"
    assert summary.dataset_drift_detected is True
    assert summary.drifted_features_count >= 2
    assert summary.share_of_drifted_features >= 0.4

    # Verify individual feature drift detection
    alt_metric = summary.drift_by_feature.get("altitude_mean_meters")
    assert alt_metric is not None
    assert alt_metric.drift_detected is True

    # Verify target drift was checked
    assert summary.target_drift_detected is not None
    assert len(html) > 500
