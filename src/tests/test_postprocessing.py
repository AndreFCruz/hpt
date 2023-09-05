"""Module to test post-hoc adjustment to meet fairness criteria.
"""

import logging
from collections.abc import Iterable
from typing import Tuple

import pytest
import numpy as np


@pytest.fixture(params=[0.05, 0.10, 0.2, 0.5, (0.3, 0.3, 0.4), (0.1, 0.2, 0.7)])
def sensitive_prevalence(request) -> Tuple[float]:
    prev = request.param
    if isinstance(prev, float):
        return (prev, 1 - prev)
    else:
        assert isinstance(prev, Iterable)
        assert sum(prev) == 1  # sanity check
        return prev


@pytest.fixture
def sensitive_attribute(
    sensitive_prevalence: Tuple[float],
    y_true: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Randomly generates sensitive attribute following a provided distribution.

    Parameters
    ----------
    sensitive_prevalence : Tuple[float]
        A tuple containing the probabilities associated with each sensitive
        attribute.
    y_true : np.ndarray
        The true labels.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        A 1-D array whose entries are the group (or sensitive attribute value)
        each sample belongs to.
    """
    num_groups = len(sensitive_prevalence)
    num_samples = len(y_true)

    return rng.choice(
        np.arange(num_groups),
        size=num_samples,
        p=sensitive_prevalence,
    )
