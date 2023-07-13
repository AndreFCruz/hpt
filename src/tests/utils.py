"""Test utils.
"""

import logging
import numpy as np


def get_metric_denominator(y_true: np.ndarray, threshold_target_metric: str) -> int:
    # Metric denominator is used to compute the tightest error bound on the target metric
    metric_denominator: int
    if threshold_target_metric == "tpr":
        metric_denominator = np.sum(y_true)  # LPs
    elif threshold_target_metric == "fpr":
        metric_denominator = np.sum(1 - y_true)  # LNs
    elif threshold_target_metric == "ppr":
        metric_denominator = len(y_true)  # all samples
    else:
        logging.warning(
            "Please implement the max error bound here if you implement new "
            "target metrics."
        )
        metric_denominator = len(y_true)

    return metric_denominator
