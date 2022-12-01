"""A set of functions to evaluate predictions on common performance
and fairness metrics, possibly at a specified FPR or FNR target.
"""

import math
import logging
from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix

from .utils.fairness_criteria import CriteriaData


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

    # Compute ratios
    for metric_name in unique_metrics:
        ratio_name = f"{metric_name}_ratio"
        curr_metric_results = [
            groupwise_metrics[group_metric_name(metric_name, group_name)]
            for group_name in unique_groups
        ]

        results[ratio_name] = safe_division(
            min(curr_metric_results),
            max(curr_metric_results)
        )

    # Optionally, return group-wise metrics as well
    if return_groupwise_metrics:
        results.update(groupwise_metrics)

    return results


def compute_binary_predictions(
        y_true: np.ndarray,
        y_pred_scores: np.ndarray,
        threshold: Optional[float] = None,
        tpr: Optional[float] = None,
        fpr: Optional[float] = None,
        ppr: Optional[int] = None,
        random_seed: Optional[int] = 42,
    ) -> np.ndarray:
    """Discretizes the given score predictions into binary labels,
    according to the provided target metric for thresholding.

    Parameters
    ----------
    y_true : np.ndarray
        The true binary labels
    y_pred_scores : np.ndarray
        Predictions as a continuous score between 0 and 1
    threshold : Optional[float], optional
        Whether to use a specified (global) threshold, by default None
    tpr : Optional[float], optional
        Whether to target a specified TPR (true positive rate, or recall), by
        default None
    fpr : Optional[float], optional
        Whether to target a specified FPR (false positive rate), by default None
    ppr : Optional[float], optional
        Whether to target a specified PPR (positive prediction rate), by default
        None

    Returns
    -------
    np.ndarray
        The binarized predictions according to the specified target.
    """
    assert sum(1 for val in {threshold, fpr, tpr, ppr} if val is not None) == 1, (
        f"Please provide exactly one of (threshold, fpr, tpr, ppr); got "
        f"{(threshold, fpr, tpr, ppr)}."
    )

    # If threshold provided, just binarize it, no untying necessary
    if threshold:
        return (y_pred_scores >= threshold).astype(int)
    
    # Otherwise, we need to compute the allowed value for the numerator
    # and corresponding threshold (plus, may require random untying)
    label_pos = np.count_nonzero(y_true)
    label_neg = np.count_nonzero(1 - y_true)
    assert (total := label_pos + label_neg) == len(y_true)  # sanity check

    # Indices of predictions ordered by score, descending
    y_pred_sorted_indices = np.argsort(-y_pred_scores)

    # Labels ordered by descending prediction score
    y_true_sorted = y_true[y_pred_sorted_indices]

    # Number of positive predictions allowed according to the given metric
    # (the allowed budget for the metric's numerator)
    positive_preds_budget: int

    # Samples that count for the positive_preds_budget
    # (LPs for TPR, LNs for FPR, and all samples for PPR)
    # (related to the metric's denominator)
    target_samples_mask: np.ndarray
    if tpr:
        # TPs budget to ensure >= the target TPR
        positive_preds_budget = math.ceil(tpr * label_pos)
        target_samples_mask = y_true_sorted == 1  # label positive samples
    
    elif fpr:
        # FPs budget to ensure <= the target FPR
        positive_preds_budget = math.floor(fpr * label_neg)
        target_samples_mask = y_true_sorted == 0  # label negative samples

    elif ppr:
        # PPs budget to ensure <= the target PPR
        positive_preds_budget = math.floor(ppr * total)
        target_samples_mask = np.ones_like(y_true_sorted).astype(bool) # all samples

    # Indices of target samples (relevant for the target metric), ordered by descending score
    target_samples_indices = y_pred_sorted_indices[target_samples_mask]

    # Find the threshold at which the specified numerator_budget is met
    threshold_idx = target_samples_indices[positive_preds_budget]
    threshold = y_pred_scores[threshold_idx]

    ####################################
    # Code for random untying follows: #
    ####################################
    y_pred_binary = (y_pred_scores >= threshold).astype(int)

    # 1. compute actual number of positive predictions (on relevant target samples)
    actual_pos_preds = np.sum(y_pred_binary[target_samples_indices])

    # 2. check if this number corresponds to the target
    if actual_pos_preds != positive_preds_budget:
        logging.warning(
            "Target metric for thresholding could not be met, will randomly "
            "untie samples with the same predicted score to fulfill target."
        )

        assert actual_pos_preds > positive_preds_budget, (
            "Sanity check: actual number of positive predictions should always "
            "be higher or equal to the target number when following this "
            f"algorithm; got actual={actual_pos_preds}, target={positive_preds_budget};"
        )

        # 2.1. if target was not met, compute number of extra predicted positives
        extra_pos_preds = actual_pos_preds - positive_preds_budget

        # 2.2. randomly select extra_pos_preds among the relevant
        # samples (either TPs or FPs or PPs) with the same score
        rng = np.random.RandomState(random_seed)

        samples_at_target_threshold_mask = (y_pred_scores[y_pred_sorted_indices] == threshold)

        target_samples_at_target_threshold_indices = (
            y_pred_sorted_indices[
                samples_at_target_threshold_mask &      # Filter for samples at target threshold
                target_samples_mask                     # Filter for relevant (target) samples
            ]
        )

        # # The extra number of positive predictions must be fully explained by this score tie
        # import ipdb; ipdb.set_trace()   # TODO: figure out why this assertion fails
        # assert extra_pos_preds < len(target_samples_at_target_threshold_indices)

        extra_pos_preds_indices = rng.choice(
            target_samples_at_target_threshold_indices,
            size=extra_pos_preds,
            replace=False)

        # 2.3. give extra_pos_preds_indices a negative prediction
        y_pred_binary[extra_pos_preds_indices] = 0


    # Sanity check: the number of positive_preds_budget should now be exactly fulfilled
    assert np.sum(y_pred_binary[target_samples_indices]) == positive_preds_budget

    return y_pred_binary


def compute_binary_predictions_posthoc_adjustment(
        y_true: np.ndarray,
        y_pred_scores: np.ndarray,
        sensitive_attribute: np.ndarray,
        equalize_tpr: Optional[bool] = False,
        equalize_fpr: Optional[bool] = False,
        false_negative_cost: Optional[float] = 1,
        false_positive_cost: Optional[float] = 1,
        # allowed_tpr_gap: Optional[float] = 0.0,
        # allowed_fpr_gap: Optional[float] = 0.0,
        random_seed: Optional[int] = 42,
    ) -> np.ndarray:
    """Discretizes the given score predictions into binary labels, according
    to the provided fairness criteria - equalize TPR, FPR, or both.

    Parameters
    ----------
    y_true : np.ndarray
        The true binary labels.
    y_pred_scores : np.ndarray
        Predictions as a continuous score between 0 and 1.
    sensitive_attribute : np.ndarray
        Sensitive attribute representing the group each sample belongs to.
    equalize_tpr : Optional[bool], optional
        Whether to equalize group-wise TPR, by default False.
    equalize_fpr : Optional[bool], optional
        Whether to equalize group-wise FPR, by default False.
    false_negative_cost : Optional[float], optional
        The cost of a false negative (FN) prediction, by default 1.
    false_positive_cost : Optional[float], optional
        The cost of a false positive (FP) prediction, by default 1.
    allowed_tpr_gap : Optional[float], optional
        The allowed gap in group-wise TPR (if any), by default 0.
    allowed_fpr_gap : Optional[float], optional
        The allowed gap in group-wise FPR (if any), by default 0.
    random_seed : Optional[int], optional
        The random seed used in case some prediction randomization is required
        (e.g.,  in the case of equal odds - equalizing both TPR and FPR).

    Returns
    -------
    np.ndarray
        The binarized predictions.
    """
    assert equalize_fpr or equalize_tpr, \
        "Must target either equal FPR or equal TPR or both, got neither."
    
    # The threshold is computed from the ratio of FPs and FNs,
    # and assumes the underlying classifier is approximately calibrated
    # i.e., a threshold of t=0.5 indicates equal likelihood of paying the FP
    # cost or the FN cost
    threshold = false_positive_cost / (false_positive_cost + false_negative_cost)

    # Generate DataFrames for CDF, Performance, and Total per group, where:
    # > cdf[group_][score_] = proportion of members of group_ with score below score_
    cdf = ...           # TODO...

    # > perf[group_][score_] = proportion of members of that group_ and score_ that have a positive label
    performance = ...   # TODO...

    # > totals[group_] = total number of group_ members
    totals = {}

    # Construct CriteriaData object to compute fairness criteria and thresholds
    data = CriteriaData(cdfs, performance, totals)

    # > groupwise_thresholds[group_] = threshold used for group_
    # TODO: check if it is 1{f(X) >= threshold} or 1{f(X) > threshold}
    groupwise_thresholds: dict
    if equalize_tpr and equalize_fpr:   # Equal odds

        # TODO
        # - use data.two_sided_optimum to get optimal Equal Odds point (under all ROC curves)
        # - somehow binarize predictions such that this is met (with randomization for a portion of predictions)

        rng = np.random.RandomState(random_seed)
        pass

    elif equalize_tpr:                  # Equal opportunity among Y=1
        groupwise_thresholds = data.opportunity_cutoffs(target_rate=threshold)
    
    elif equalize_fpr:                  # Equal opportunity among Y=0
        raise NotImplementedError(
            "Equalizing TNR or FPR is not yet implemented as a fairness "
            "criterion."
        )
    
    # Binarize predictions with the given thresholds
    y_pred_binary = np.zeros_like(y_true)

    unique_groups = np.unique(sensitive_attribute)
    for group_ in unique_groups:
        group_mask = sensitive_attribute == group_
        y_pred_binary[group_mask] = (
            y_pred_scores[group_mask] >= groupwise_thresholds[group_]
        ).astype(int)
    
    return y_pred_binary
