"""Unit tests for the evaluation metrics module."""

import numpy as np
import pandas as pd
import pytest

from coffee_quality.models.evaluator import EvaluationMetrics, evaluate_predictions


def test_evaluate_predictions_perfect():
    """Test metrics with 100% correct predictions."""
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_prob = [0.1, 0.2, 0.8, 0.9]

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.macro_f1 == 1.0
    assert metrics.precision_commercial == 1.0
    assert metrics.recall_commercial == 1.0
    assert metrics.f1_commercial == 1.0
    assert metrics.precision_specialty == 1.0
    assert metrics.recall_specialty == 1.0
    assert metrics.f1_specialty == 1.0
    assert metrics.true_negatives == 2
    assert metrics.true_positives == 2
    assert metrics.false_positives == 0
    assert metrics.false_negatives == 0
    assert metrics.confusion_matrix == [[2, 0], [0, 2]]

    d = metrics.to_dict()
    assert d["accuracy"] == 1.0
    assert d["confusion_matrix"] == [[2, 0], [0, 2]]


def test_evaluate_predictions_inverted():
    """Test metrics with completely inverted predictions."""
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    y_prob = [0.9, 0.8, 0.2, 0.1]

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert metrics.accuracy == 0.0
    assert metrics.roc_auc == 0.0
    assert metrics.macro_f1 == 0.0
    assert metrics.true_negatives == 0
    assert metrics.true_positives == 0
    assert metrics.false_positives == 2
    assert metrics.false_negatives == 2


def test_evaluate_predictions_without_probabilities():
    """Test metrics calculation when probabilities are omitted."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]

    metrics = evaluate_predictions(y_true, y_pred)

    assert metrics.accuracy == 0.75
    assert metrics.roc_auc is None
    assert metrics.true_negatives == 1
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.true_positives == 2


def test_evaluate_predictions_single_class_target():
    """Test that roc_auc is safely None when y_true has only one class."""
    y_true = [1, 1, 1, 1]
    y_pred = [1, 1, 1, 1]
    y_prob = [0.7, 0.8, 0.85, 0.9]

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc is None


def test_evaluate_predictions_imbalanced_naive_baseline():
    """Verify that a naive all-specialty predictor exposes low macro-F1 despite high accuracy."""
    # 86 specialty (1) and 14 commercial (0)
    y_true = [1] * 86 + [0] * 14
    y_pred = [1] * 100  # Naive predictor
    y_prob = [0.9] * 100

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert metrics.accuracy == 0.86
    assert metrics.recall_commercial == 0.0
    assert metrics.precision_commercial == 0.0
    assert metrics.f1_commercial == 0.0
    assert metrics.recall_specialty == 1.0
    assert metrics.macro_f1 < 0.50  # Strongly penalizes failing the minority class


def test_evaluate_predictions_pandas_and_numpy_inputs():
    """Test compatibility with pandas Series and numpy arrays."""
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    y_prob = pd.Series([0.2, 0.8, 0.1, 0.9])

    metrics = evaluate_predictions(y_true, y_pred, y_prob)

    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.accuracy == 1.0
    assert metrics.roc_auc == 1.0


def test_evaluate_predictions_validation_errors():
    """Test validation errors for invalid shapes, types, and values."""
    # Empty inputs
    with pytest.raises(ValueError, match="Cannot evaluate empty arrays"):
        evaluate_predictions([], [])

    # 2D arrays
    with pytest.raises(ValueError, match="must be 1-dimensional"):
        evaluate_predictions([[0, 1]], [[0, 1]])

    # Length mismatch true vs pred
    with pytest.raises(ValueError, match="Length mismatch: y_true has 2 samples, but y_pred has 3 samples"):
        evaluate_predictions([0, 1], [0, 1, 1])

    # Length mismatch true vs prob
    with pytest.raises(ValueError, match="Length mismatch: y_prob has 3 samples, but y_true has 2 samples"):
        evaluate_predictions([0, 1], [0, 1], [0.1, 0.2, 0.3])

    # 2D probabilities
    with pytest.raises(ValueError, match="y_prob must be a 1-dimensional array"):
        evaluate_predictions([0, 1], [0, 1], [[0.1, 0.9], [0.2, 0.8]])

    # Invalid labels in y_true
    with pytest.raises(ValueError, match="y_true contains invalid labels"):
        evaluate_predictions([0, 2], [0, 1])

    # Invalid labels in y_pred
    with pytest.raises(ValueError, match="y_pred contains invalid labels"):
        evaluate_predictions([0, 1], [0, -1])

    # Probabilities out of [0, 1] bounds
    with pytest.raises(ValueError, match=r"y_prob values must be in the range \[0.0, 1.0\]"):
        evaluate_predictions([0, 1], [0, 1], [-0.1, 0.5])

    with pytest.raises(ValueError, match=r"y_prob values must be in the range \[0.0, 1.0\]"):
        evaluate_predictions([0, 1], [0, 1], [0.5, 1.2])

    # Probabilities contain NaN
    with pytest.raises(ValueError, match="y_prob contains NaN values"):
        evaluate_predictions([0, 1], [0, 1], [0.5, float("nan")])
