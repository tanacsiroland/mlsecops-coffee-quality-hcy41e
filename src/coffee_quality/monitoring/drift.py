"""EvidentlyAI-powered data and target drift monitoring engine.

Detects statistical distribution shifts in Arabica coffee lot attributes
by comparing live inference audit samples against the training baseline reference.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from typing import Any, cast
import warnings

import numpy as np
import pandas as pd
import pandas.core.dtypes.common as _pd_common


def _compat_infer_dtype(x: Any) -> Any:
    return getattr(x, "dtype", np.dtype(object))


# Pandas 3.x compatibility shim for Evidently 0.7.x legacy tests import
if not hasattr(_pd_common, "infer_dtype_from_object"):
    setattr(_pd_common, "infer_dtype_from_object", _compat_infer_dtype)

from evidently import Report  # noqa: E402
from evidently.presets import DataDriftPreset  # noqa: E402

from coffee_quality.config import Settings, get_settings  # noqa: E402
from coffee_quality.data.cleaner import clean_dataset  # noqa: E402
from coffee_quality.data.loader import load_raw_data  # noqa: E402

logger = logging.getLogger(__name__)

PREDICTIVE_COLUMNS: tuple[str, ...] = (
    "Country.of.Origin",
    "Variety",
    "Processing.Method",
    "Color",
    "altitude_mean_meters",
    "Moisture",
    "Category.One.Defects",
    "Category.Two.Defects",
    "Quakers",
    "Grading.Date",
)

TARGET_COLUMN = "Is_Specialty"


@dataclass(frozen=True)
class FeatureDriftMetric:
    """Detailed statistical drift metrics for a single feature column."""

    column_name: str
    method: str
    threshold: float
    score: float
    drift_detected: bool


@dataclass(frozen=True)
class DriftReportSummary:
    """Consolidated summary of data and target drift detection."""

    dataset_drift_detected: bool
    drifted_features_count: int
    total_features_count: int
    share_of_drifted_features: float
    drift_by_feature: dict[str, FeatureDriftMetric]
    target_drift_detected: bool | None
    reference_sample_count: int
    current_sample_count: int
    timestamp_utc: str
    status: str = "success"

    def to_dict(self) -> dict[str, Any]:
        """Convert report summary to serializable dictionary."""
        return asdict(self)


class DriftDetector:
    """Automated data and concept drift detection engine."""

    def __init__(
        self,
        reference_data: pd.DataFrame,
        drift_share_threshold: float = 0.5,
        min_current_samples: int = 5,
    ) -> None:
        """Initialize DriftDetector with reference dataset and drift thresholds.

        Args:
            reference_data: Baseline reference DataFrame.
            drift_share_threshold: Proportion of features that must drift to trigger dataset drift (default 0.5).
            min_current_samples: Minimum number of live samples required for drift calculation (default 5).
        """
        self.reference_data = reference_data.copy()
        self.drift_share_threshold = drift_share_threshold
        self.min_current_samples = min_current_samples

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        drift_share_threshold: float = 0.5,
        min_current_samples: int = 5,
    ) -> DriftDetector:
        """Instantiate DriftDetector loading reference baseline dataset from configuration.

        Args:
            settings: Optional Settings override.
            drift_share_threshold: Proportion of features to trigger dataset drift.
            min_current_samples: Minimum required live samples.

        Returns:
            Configured DriftDetector instance.
        """
        cfg = settings or get_settings()
        raw_df, _ = load_raw_data(cfg.data_raw_path)
        clean_df = clean_dataset(raw_df)
        return cls(
            reference_data=clean_df,
            drift_share_threshold=drift_share_threshold,
            min_current_samples=min_current_samples,
        )

    def calculate_drift(
        self,
        current_data: pd.DataFrame,
    ) -> tuple[DriftReportSummary, str]:
        """Execute statistical drift tests comparing current data against reference data.

        Args:
            current_data: Live inference samples DataFrame.

        Returns:
            Tuple containing structured DriftReportSummary and rendered HTML report string.
        """
        now_utc = datetime.now(timezone.utc).isoformat()
        current_count = len(current_data)
        ref_count = len(self.reference_data)

        # Insufficient sample guard
        if current_count < self.min_current_samples:
            logger.warning(
                "Insufficient current samples for drift detection: %d provided, %d required.",
                current_count,
                self.min_current_samples,
            )
            summary = DriftReportSummary(
                dataset_drift_detected=False,
                drifted_features_count=0,
                total_features_count=0,
                share_of_drifted_features=0.0,
                drift_by_feature={},
                target_drift_detected=None,
                reference_sample_count=ref_count,
                current_sample_count=current_count,
                timestamp_utc=now_utc,
                status="insufficient_samples",
            )
            html_msg = (
                f"<html><body><h3>Insufficient Samples</h3>"
                f"<p>Provided {current_count} samples. At least {self.min_current_samples} "
                f"samples are required to compute statistical drift.</p></body></html>"
            )
            return summary, html_msg

        # Filter common columns
        target_cols = [c for c in PREDICTIVE_COLUMNS if c in self.reference_data.columns and c in current_data.columns]
        if TARGET_COLUMN in self.reference_data.columns and TARGET_COLUMN in current_data.columns:
            target_cols.append(TARGET_COLUMN)

        ref_subset = cast(pd.DataFrame, self.reference_data[target_cols].copy())
        cur_subset = cast(pd.DataFrame, current_data[target_cols].copy())

        report = Report(metrics=[DataDriftPreset(drift_share=self.drift_share_threshold)])

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            snapshot = report.run(reference_data=ref_subset, current_data=cur_subset)

        html_report = snapshot.get_html_str(as_iframe=False)
        snapshot_dict = snapshot.dict()

        drift_by_feature: dict[str, FeatureDriftMetric] = {}
        target_drift_detected: bool | None = None
        dataset_drift_detected = False
        drifted_count = 0
        total_count = 0
        drift_share = 0.0

        for metric in snapshot_dict.get("metrics", []):
            metric_type = metric.get("config", {}).get("type", "")

            # Overall dataset drift metric
            if "DriftedColumnsCount" in metric_type or "DriftedColumnsCount" in metric.get("metric_name", ""):
                val = metric.get("value", {})
                if isinstance(val, dict):
                    drifted_count = int(val.get("count", 0))
                    drift_share = round(float(val.get("share", 0.0)), 4)
                    dataset_drift_detected = drift_share >= self.drift_share_threshold

            # Per-feature value drift metric
            elif "ValueDrift" in metric_type or "ValueDrift" in metric.get("metric_name", ""):
                cfg = metric.get("config", {})
                col_name = str(cfg.get("column", ""))
                method = str(cfg.get("method", ""))
                threshold = float(cfg.get("threshold", 0.05))
                score = round(float(metric.get("value", 0.0)), 4)

                # Determine if drift is detected based on testing methodology
                if "p_value" in method.lower():
                    drift_detected = score < threshold
                else:
                    drift_detected = score >= threshold

                if col_name == TARGET_COLUMN:
                    target_drift_detected = drift_detected
                elif col_name in PREDICTIVE_COLUMNS:
                    drift_by_feature[col_name] = FeatureDriftMetric(
                        column_name=col_name,
                        method=method,
                        threshold=threshold,
                        score=score,
                        drift_detected=drift_detected,
                    )

        total_count = len(drift_by_feature)
        if total_count > 0:
            drifted_feature_count = sum(1 for m in drift_by_feature.values() if m.drift_detected)
            actual_drift_share = round(drifted_feature_count / total_count, 4)
            dataset_drift_detected = actual_drift_share >= self.drift_share_threshold
        else:
            drifted_feature_count = drifted_count
            actual_drift_share = drift_share

        summary = DriftReportSummary(
            dataset_drift_detected=dataset_drift_detected,
            drifted_features_count=drifted_feature_count,
            total_features_count=total_count,
            share_of_drifted_features=actual_drift_share,
            drift_by_feature=drift_by_feature,
            target_drift_detected=target_drift_detected,
            reference_sample_count=ref_count,
            current_sample_count=current_count,
            timestamp_utc=now_utc,
            status="success",
        )

        logger.info(
            "Drift analysis complete: %d/%d features drifted (share: %.2f, dataset_drift: %s).",
            drifted_feature_count,
            total_count,
            actual_drift_share,
            dataset_drift_detected,
        )

        return summary, html_report
