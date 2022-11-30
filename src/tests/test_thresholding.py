"""Module to test thresholding functions.
"""

import pytest
import numpy as np
from .utils import get_metric_denominator

# Number of samples used for testing
NUM_SAMPLES = 10_000
RANDOM_SEED = 42


@pytest.fixture
def rng():
    return np.random.RandomState(RANDOM_SEED)


@pytest.fixture(params=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
def prevalence(request) -> float:
    return request.param


@pytest.fixture
def y_true(prevalence: float, rng: np.random.RandomState) -> np.ndarray:
    assert 0 <= prevalence <= 1
    # Generate labels
    return (rng.random(NUM_SAMPLES) <= prevalence).astype(int)


@pytest.fixture
def y_pred_scores(rng: np.random.RandomState) -> np.ndarray:
    return rng.random(NUM_SAMPLES)


@pytest.fixture(params=[2, 5, 10, 100, 1000, 10000])
def num_score_bins(request) -> int:
    return request.param


@pytest.fixture
def y_pred_scores_with_ties(num_score_bins: int, rng: np.random.RandomState) -> np.ndarray:
    """Randomly generates score predictions with ties.

    NOTE
    - I know there's a bit of confusion with the num_score_bins because to
    get scores evenly spaced by 0.1 we need 11 buckets not 10;
    - This works just fine as each bin gets all scores within range 
    (bin_score - 1/num_score_bins/2, bin_score + 1/num_score_bins/2);
    - For bins 0.0 and 1.0 this range is halved as there are no negative scores
    or scores above 1.0;
    - All in all, it works just fine...


    Parameters
    ----------
    num_score_bins : int
        Number of bins used to discretize scores.

    rng : np.random.RandomState
        Random number generator.

    Returns
    -------
    np.ndarray
        Returns score predictions with ties.
    """
    # Discretized scores distributed uniformly at random in range [0, n_buckets]
    scores = ((rng.random(NUM_SAMPLES) + 0.05) * num_score_bins).astype(int)
    return (scores / num_score_bins).clip(0.0, 1.0)


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
    ):
    """Tests hpt package score binarization (without score ties).
    """

    # Binarize predictions
    from hpt.evaluation import compute_binary_predictions
    y_pred_binary = compute_binary_predictions(
        y_true, y_pred_scores,
        random_seed=RANDOM_SEED,
        **{threshold_target_metric: threshold_target_metric_value},
    )

    # Evaluate predictions
    from hpt.evaluation import evaluate_performance
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
    ):
    """Tests hpt package score binarization **with** score ties!
    """
    # Binarize predictions
    from hpt.evaluation import compute_binary_predictions
    y_pred_binary = compute_binary_predictions(
        y_true, y_pred_scores_with_ties,
        random_seed=RANDOM_SEED,
        **{threshold_target_metric: threshold_target_metric_value},
    )

    # Evaluate predictions
    from hpt.evaluation import evaluate_performance
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
