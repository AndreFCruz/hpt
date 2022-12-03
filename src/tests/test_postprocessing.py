"""Module to test post-hoc adjustment to meet fairness criteria.
"""

import pytest
import numpy as np

from hpt.evaluation import (
    evaluate_performance,
    evaluate_fairness,
)
from hpt.binarize import compute_binary_predictions_posthoc_adjustment


def test_scores_binarization_equal_tpr(
        y_true: np.ndarray,
        y_pred_scores_with_ties: np.ndarray,
        random_seed: int,
    ):
    assert True
