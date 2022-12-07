"""A set of functions to evaluate predictions on common performance
and fairness metrics, possibly at a specified FPR or FNR target.
"""

from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix

from .binarize import compute_binary_predictions


def safe_division(a: float, b: float):
    return 0 if b == 0 else a / b


def evaluate_performance(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Evaluates the provided predictions on common performance metrics.
    NOTE: currently assumes labels and predictions are binary - should we extend
    it to multi-class labels?

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The discretized predictions.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=(0, 1)).ravel()

    total = tn + fp + fn + tp
    pred_pos = tp + fp
    pred_neg = tn + fn
    assert pred_pos + pred_neg == total

    label_pos = tp + fn
    label_neg = tn + fp
    assert label_pos + label_neg == total

    results = {}
    
    # Accuracy
    results["accuracy"] = (tp + tn) / total

    # True Positive Rate (Recall)
    results["tpr"] = safe_division(tp, label_pos)

    # False Positive Rate
    results["fpr"] = safe_division(fp, label_neg)

    # True Negative Rate
    # results["tnr"] = tn / label_neg

    # Precision
    results["precision"] = safe_division(tp, pred_pos)

    # Positive Prediction Rate
    results["ppr"] = safe_division(pred_pos, total)

    return results


def evaluate_fairness(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_attribute: np.ndarray,
        return_groupwise_metrics: Optional[bool] = False,
    ) -> dict:
    """Evaluates fairness as the ratios between group-wise performance metrics.

    Parameters
    ----------
    y_true : np.ndarray
        The true class labels.
    y_pred : np.ndarray
        The discretized predictions.
    sensitive_attribute : np.ndarray
        The sensitive attribute (protected group membership).
    return_groupwise_metrics : Optional[bool], optional
        Whether to return group-wise performance metrics (bool: True) or only
        the ratios between these metrics (bool: False), by default False.

    Returns
    -------
    dict
        A dictionary with key-value pairs of (metric name, metric value).
    """
    # All unique values for the sensitive attribute
    unique_groups = np.unique(sensitive_attribute)

    results = {}
    groupwise_metrics = {}
    unique_metrics = set()

    # Helper to compute key/name of a group-wise metric
    def group_metric_name(metric_name, group_name):
        return f"{metric_name}_group={group_name}"

    assert len(unique_groups) > 1, (
        f"Found a single unique sensitive attribute: {unique_groups}")

    for s_value in unique_groups:
        # Indices of samples that belong to the current group
        group_indices = np.argwhere(sensitive_attribute == s_value).flatten()

        # Filter labels and predictions for samples of the current group
        group_labels = y_true[group_indices]
        group_preds = y_pred[group_indices]

        # Evaluate group-wise performance
        curr_group_metrics = evaluate_performance(group_labels, group_preds)

        # Add group-wise metrics to the dictionary
        groupwise_metrics.update({
            group_metric_name(metric_name, s_value): metric_value
            for metric_name, metric_value in curr_group_metrics.items()
        })

        unique_metrics = unique_metrics.union(curr_group_metrics.keys())

    # Compute ratios and absolute diffs
    for metric_name in unique_metrics:
        curr_metric_results = [
            groupwise_metrics[group_metric_name(metric_name, group_name)]
            for group_name in unique_groups
        ]

        # Metrics' ratio
        ratio_name = f"{metric_name}_ratio"

        # NOTE: should this ratio be computed w.r.t. global performance?
        # - i.e., min(curr_metric_results) / global_curr_metric_result;
        # - same question for the absolute diff calculations;
        results[ratio_name] = safe_division(
            min(curr_metric_results),
            max(curr_metric_results)
        )

        # Metrics' absolute difference
        diff_name = f"{metric_name}_diff"
        results[diff_name] = max(curr_metric_results) - min(curr_metric_results)

    
    # Equal odds: maximum constraint violation for TPR and FPR equality
    results["equal_odds_ratio"] = min(
        results["tpr_ratio"],           # why not FNR ratio here?
        results["fpr_ratio"],           # why not TNR ratio here?
    )

    results["equal_odds_diff"] = min(
        results["tpr_diff"],            # same as FNR diff
        results["fpr_diff"],            # same as TNR diff
    )

    # Optionally, return group-wise metrics as well
    if return_groupwise_metrics:
        results.update(groupwise_metrics)

    return results


def evaluate_predictions(
        y_true: np.ndarray,
        y_pred_scores: np.ndarray,
        sensitive_attribute: Optional[np.ndarray] = None,
        **threshold_target,
    ) -> dict:
    """Evaluates the given predictions on both performance and fairness
    metrics (if `sensitive_attribute` is provided).

    Parameters
    ----------
    y_true : np.ndarray
        The true labels.
    y_pred_scores : np.ndarray
        The predicted scores.
    sensitive_attribute : np.ndarray
        The sensitive attribute - which protected group each sample belongs
        to.

    Returns
    -------
    dict
        A dictionary of (key, value) -> (metric_name, metric_value).
    """

    # Binarize predictions according to the given threshold target
    y_pred_binary = compute_binary_predictions(
        y_true, y_pred_scores, **threshold_target,
    )

    # Compute global performance metrics
    results = evaluate_performance(y_true, y_pred_binary)

    # (Optionally) Compute fairness metrics
    if sensitive_attribute is not None:
        results.update(evaluate_fairness(
            y_true, y_pred_binary, sensitive_attribute,
            return_groupwise_metrics=False,
        ))

    return results
