"""Module to test thresholding functions.
"""

import pytest
import numpy as np

from hpt.evaluation import evaluate_performance
from hpt.binarize import compute_binary_predictions

from .utils import get_metric_denominator


@pytest.fixture(params=["tpr", "fpr", "ppr"])
def threshold_target_metric(request) -> str:
    return request.param


@pytest.fixture(params=[0.05, 0.10, 0.25, 0.50])
def threshold_target_metric_value(request) -> float:
    return request.param


def test_scores_binarization(
    y_true: np.ndarray,
    y_pred_scores: np.ndarray,
    threshold_target_metric: str,
    threshold_target_metric_value: float,
    random_seed: int,
):
    """Tests hpt package score binarization (without score ties)."""

    # Binarize predictions
    y_pred_binary = compute_binary_predictions(
        y_true,
        y_pred_scores,
        random_seed=random_seed,
        **{threshold_target_metric: threshold_target_metric_value},
    )

    # Evaluate predictions
    results = evaluate_performance(y_true, y_pred_binary)

    # Assert target metric is approximately met
    assert np.isclose(
        threshold_target_metric_value,
        results[threshold_target_metric],
        rtol=0.0,
        atol=1 / get_metric_denominator(y_true, threshold_target_metric),
        # ^ tolerance of at most one mistake over relevant samples
    ), (
        f"Targeting {threshold_target_metric_value:.3%} {threshold_target_metric.upper()}, "
        f"got {results[threshold_target_metric]:.3%}"
    )


def test_scores_binarization_with_ties(
    y_true: np.ndarray,
    y_pred_scores_with_ties: np.ndarray,
    threshold_target_metric: str,
    threshold_target_metric_value: float,
    random_seed: int,
):
    """Tests hpt package score binarization **with** score ties!"""
    # Binarize predictions
    y_pred_binary = compute_binary_predictions(
        y_true,
        y_pred_scores_with_ties,
        random_seed=random_seed,
        **{threshold_target_metric: threshold_target_metric_value},
    )

    # Evaluate predictions
    results = evaluate_performance(y_true, y_pred_binary)

    # Assert target metric is approximately met
    assert np.isclose(
        threshold_target_metric_value,
        results[threshold_target_metric],
        rtol=0.0,
        atol=1 / get_metric_denominator(y_true, threshold_target_metric),
        # ^ tolerance of at most one mistake over relevant samples
    ), (
        f"Targeting {threshold_target_metric_value:.3%} {threshold_target_metric.upper()}, "
        f"got {results[threshold_target_metric]:.3%}"
    )
