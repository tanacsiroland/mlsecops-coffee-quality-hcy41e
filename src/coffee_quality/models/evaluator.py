"""Evaluation metrics computation for Arabica coffee quality binary classification."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence, cast

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class EvaluationMetrics:
    """Metrics container for binary classification evaluation.

    Attributes:
        accuracy: Proportion of correct predictions overall.
        roc_auc: Area Under the ROC Curve (None if probabilities unavailable or single class).
        macro_f1: Unweighted average F1-score across commercial (0) and specialty (1).
        precision_commercial: Precision for class 0 (Commercial).
        recall_commercial: Recall for class 0 (Commercial).
        f1_commercial: F1-score for class 0 (Commercial).
        precision_specialty: Precision for class 1 (Specialty).
        recall_specialty: Recall for class 1 (Specialty).
        f1_specialty: F1-score for class 1 (Specialty).
        true_negatives: Count of correctly identified commercial coffees (TN).
        false_positives: Count of commercial coffees incorrectly predicted as specialty (FP).
        false_negatives: Count of specialty coffees incorrectly predicted as commercial (FN).
        true_positives: Count of correctly identified specialty coffees (TP).
    """

    accuracy: float
    roc_auc: float | None
    macro_f1: float
    precision_commercial: float
    recall_commercial: float
    f1_commercial: float
    precision_specialty: float
    recall_specialty: float
    f1_specialty: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """Return 2x2 confusion matrix [[TN, FP], [FN, TP]]."""
        return [
            [self.true_negatives, self.false_positives],
            [self.false_negatives, self.true_positives],
        ]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to a dictionary suitable for logging and serialization."""
        data = asdict(self)
        data["confusion_matrix"] = self.confusion_matrix
        return data


def evaluate_predictions(
    y_true: Sequence[int] | np.ndarray[Any, Any] | pd.Series,
    y_pred: Sequence[int] | np.ndarray[Any, Any] | pd.Series,
    y_prob: Sequence[float] | np.ndarray[Any, Any] | pd.Series | None = None,
) -> EvaluationMetrics:
    """Compute comprehensive evaluation metrics for binary predictions.

    Validates that inputs are 1D arrays of matching lengths with binary (0 or 1) values.
    Probabilities in y_prob, if provided, must lie in the range [0.0, 1.0].

    Args:
        y_true: Ground-truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_prob: Optional predicted probabilities for the positive class (1 = Specialty).

    Returns:
        EvaluationMetrics instance containing computed metric values.

    Raises:
        ValueError: If array lengths mismatch, arrays are empty, values are outside
            binary bounds, or probability values are invalid.
    """
    true_arr = np.asarray(y_true)
    pred_arr = np.asarray(y_pred)

    if true_arr.ndim != 1 or pred_arr.ndim != 1:
        raise ValueError("y_true and y_pred must be 1-dimensional arrays.")

    if len(true_arr) == 0:
        raise ValueError("Cannot evaluate empty arrays.")

    if len(true_arr) != len(pred_arr):
        raise ValueError(
            f"Length mismatch: y_true has {len(true_arr)} samples, but y_pred has {len(pred_arr)} samples."
        )

    valid_labels = {0, 1}
    unique_true = set(np.unique(true_arr).tolist())
    if not unique_true.issubset(valid_labels):
        raise ValueError(f"y_true contains invalid labels {unique_true - valid_labels}. Allowed: {valid_labels}.")

    unique_pred = set(np.unique(pred_arr).tolist())
    if not unique_pred.issubset(valid_labels):
        raise ValueError(f"y_pred contains invalid labels {unique_pred - valid_labels}. Allowed: {valid_labels}.")

    prob_arr: np.ndarray[Any, Any] | None = None
    if y_prob is not None:
        prob_arr = np.asarray(y_prob)
        if prob_arr.ndim != 1:
            raise ValueError("y_prob must be a 1-dimensional array.")
        if len(prob_arr) != len(true_arr):
            raise ValueError(
                f"Length mismatch: y_prob has {len(prob_arr)} samples, but y_true has {len(true_arr)} samples."
            )
        if np.any(np.isnan(prob_arr)):
            raise ValueError("y_prob contains NaN values.")
        if np.any((prob_arr < 0.0) | (prob_arr > 1.0)):
            raise ValueError("y_prob values must be in the range [0.0, 1.0].")

    acc = float(accuracy_score(true_arr, pred_arr))
    macro_f1 = float(f1_score(true_arr, pred_arr, average="macro", zero_division=cast(Any, 0)))

    precision_raw, recall_raw, f1_raw, _ = cast(
        tuple[np.ndarray[Any, Any], np.ndarray[Any, Any], np.ndarray[Any, Any], Any],
        precision_recall_fscore_support(true_arr, pred_arr, labels=[0, 1], zero_division=cast(Any, 0)),
    )

    prec_comm, prec_spec = float(precision_raw[0]), float(precision_raw[1])
    rec_comm, rec_spec = float(recall_raw[0]), float(recall_raw[1])
    f1_comm, f1_spec = float(f1_raw[0]), float(f1_raw[1])

    cm = confusion_matrix(true_arr, pred_arr, labels=[0, 1])
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])

    roc_auc: float | None = None
    if prob_arr is not None:
        if len(unique_true) > 1:
            roc_auc = float(roc_auc_score(true_arr, prob_arr))

    return EvaluationMetrics(
        accuracy=acc,
        roc_auc=roc_auc,
        macro_f1=macro_f1,
        precision_commercial=prec_comm,
        recall_commercial=rec_comm,
        f1_commercial=f1_comm,
        precision_specialty=prec_spec,
        recall_specialty=rec_spec,
        f1_specialty=f1_spec,
        true_negatives=tn,
        false_positives=fp,
        false_negatives=fn,
        true_positives=tp,
    )
