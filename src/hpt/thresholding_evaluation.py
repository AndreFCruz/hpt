"""Set of util functions to generate group-wise thresholds that enforce
some fairness criteria (or even just to maximize global performance).
"""


import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np

from tqdm.auto import tqdm
from sklearn.metrics import roc_curve
from sklearn.model_selection import ParameterGrid

import seaborn as sns
from matplotlib import pyplot as plt


def group_key(group):
    """Util function to compute a unique group identifier from its sensitive
    attribute value.
    """
    return str(group)


class ThresholdingEvaluation:

    def __init__(self, y_true, s_true) -> None:
        self.y_true = y_true
        self.s_true = s_true

        # Statistics for evaluation data
        self.total_samples = len(y_true)
        self.unique_groups = np.unique(s_true)

        # Number of samples per group
        self.group_sizes = {group_key(g): np.sum(s_true == g) for g in self.unique_groups}

        # Number of Label Positive samples per group
        self.group_lps = {group_key(g): np.sum(y_true[s_true == g]) for g in self.unique_groups}

        # Number of Label Negative samples per group
        self.group_lns = {group_key(g): np.sum((1 - y_true)[s_true == g]) for g in self.unique_groups}

        # sanity check
        assert self.total_samples == sum(self.group_sizes.values())
        assert self.total_samples == sum(self.group_lps.values()) + sum(self.group_lns.values())

    def compute_global_accuracy(self, groupwise_fpr: dict, groupwise_tpr: dict) -> float:
        """Computes global accuracy from groupwise FPR and TPR metrics.
        
        Parameters
        ----------
        groupwise_fpr : dict
            A dictionary in which groupwise_fpr[group_] = FPR of group group_.
        groupwise_tpr : dict
            A dictionary in which groupwise_tpr[group_] = TPR of group group_.
        
        Returns
        -------
        float
            The value for global accuracy, between 0.0 and 1.0.
        """
        total_tps, total_tns = 0, 0
        
        for g in self.unique_groups:
            # Compute TNs for group g
            # TN = TNR * LN = (1-FPR) * LN
            group_tns = (
                (1-groupwise_fpr[group_key(g)]) *
                self.group_lns[group_key(g)]
            )
            total_tns += group_tns

            # Compute TPs for group g
            group_tps = (
                groupwise_tpr[group_key(g)] *
                self.group_lps[group_key(g)]
            )
            total_tps += group_tps

        return (total_tps + total_tns) / self.total_samples

    def post_hoc_fairness(
            self,
            y_pred_scores: np.ndarray,
            equal_fpr: bool = True,
            equal_tpr: bool = True,
            fpr_tolerance: float = 1e-4,
            tpr_tolerance: float = 1e-4,
            n_thresholds: int = 100,
            n_jobs: int = 1,
            show_progress: bool = False,
            plot_rocs: bool = False,
        ):
        if equal_fpr is None and equal_tpr is None:
            logging.warning("Maximizing global accuracy with no fairness criteria enforced.")

        unique_groups = np.unique(self.s_true)
        roc_curves = dict()

        for g in unique_groups:
            group_mask = self.s_true == g
            group_y_true = self.y_true[group_mask]
            group_y_scores = y_pred_scores[group_mask]
            
            # This computes a triplet of (FPRs, TPRs, thresholds)
            fpr, tpr, thresholds = roc_curve(group_y_true, group_y_scores)
            
            # Keep only n thresholds
            keep_indices = [i * len(thresholds) // n_thresholds for i in range(n_thresholds)]
            fpr, tpr, thresholds = fpr[keep_indices], tpr[keep_indices], thresholds[keep_indices]

            roc_curves[group_key(g)] = (fpr, tpr, thresholds)
            
            if plot_rocs:
                sns.lineplot(x=fpr, y=tpr, label=f"group={g}")

        # TODO: add randomized thresholds in the future
        search_space = {
            group_key(g): [(fprs[i], tprs[i], thresholds[i]) for i in range(len(thresholds))]
            for g, (fprs, tprs, thresholds) in roc_curves.items()
        }
        
        param_iterator = ParameterGrid(search_space)
        if show_progress:
            progress_bar = tqdm(total=len(param_iterator), desc="Progress", position=0)

        def accuracy_helper(params: dict):
            unique_group_keys = {group_key(g) for g in unique_groups}
            groupwise_fpr = {g: params[g][0] for g in unique_group_keys}
            groupwise_tpr = {g: params[g][1] for g in unique_group_keys}
            # groupwise_threshold = {g: params[g][2] for g in unique_group_keys}

            if show_progress:
                progress_bar.update()

            # Check if fairness criteria is fulfilled
            fpr_disp = max(groupwise_fpr.values()) - min(groupwise_fpr.values())
            tpr_disp = max(groupwise_tpr.values()) - min(groupwise_tpr.values())

            if equal_fpr and fpr_disp > fpr_tolerance:
                acc = float("nan")  # does not fulfill FPR equality

            elif equal_tpr and tpr_disp > tpr_tolerance:
                acc = float("nan")  # does not fulfill TPR equality

            # TODO: account for randomized thresholds here...
            else:
            # If criteria are fulfilled, compute global accuracy for these parameters
                acc = self.compute_global_accuracy(
                            groupwise_fpr=groupwise_fpr,
                            groupwise_tpr=groupwise_tpr,
                        )

            return (acc, (fpr_disp, tpr_disp), params)


    #     with ProcessPoolExecutor(n_jobs) as executor:
        with ThreadPoolExecutor(n_jobs) as executor:
            accuracies = executor.map(accuracy_helper, param_iterator)
        
        accuracies = list(accuracies)

        # Select point that maximizes accuracy while fulfilling fairness criteria
        max_accu_result = max(accuracies, key=lambda t: t[0])
        
        if plot_rocs:
            _, _, params = max_accu_result
            
            for g, marker in zip(unique_groups, ['^', 'P', 'p', 'X']):
                point = params[group_key(g)]
                sns.scatterplot(
                    x=[point[0]], y=[point[1]],
                    marker=marker, s=80,
                    label=f"$t_{group_key(g)}={point[2]:.3}$",
                )

            plt.xlabel('FPR')
            plt.ylabel('TPR')
            plt.title('Group-wise ROC curves')
            
            plt.show()

        return max_accu_result
