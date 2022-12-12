"""A simple wrapper for optuna hyperparameter tuners.
"""

import re
import logging
import dataclasses
from pathlib import Path
from functools import partial
from inspect import signature
from typing import Callable, Optional, Union, List

import pandas as pd
import numpy as np
import optuna
from optuna import samplers
from optuna.trial import BaseTrial

from .suggest import suggest_callable_hyperparams
from .evaluation import (
    evaluate_performance,
    evaluate_fairness,
    evaluate_predictions,
)

from .utils.api import BaseLearner
from .utils.load_yaml import load_hyperparameter_space
from .utils.classpath import import_object


class ObjectiveFunction:
    """Callable objective function to be used with optuna.
    """

    @dataclasses.dataclass
    class TrialResults:
        id: int
        hyperparameters: dict
        validation_results: dict
        train_results: dict = None
        model: BaseLearner = None
        algorithm: str = None

        def __post_init__(self):
            match = re.match(
                r'^(?:.+[.])?(?P<algorithm>\w+)$',
                self.hyperparameters['classpath'])

            self.algorithm = match['algorithm']
    
    @staticmethod
    def instantiate_model(classpath: str, **hyperparams) -> BaseLearner:
        """Instantiates a model using the provided class path and provided
        hyperparameters.

        Parameters
        ----------
        classpath : str
            The classpath for importing the model's constructor/class.
        hyperparams : dict
            A dictionary of hyperparameter values to use as key-word arguments.

        Returns
        -------
        object
            The model object.
        """
        constructor = import_object(classpath)
        assert callable(constructor)

        # Use only arguments that are accepted by the model constructor
        # TODO: fix this, currently it's throwing away useful kwargs...
        # hyperparams_fitted = fit_dict(hyperparams, constructor)

        hyperparams_fitted = hyperparams.copy() # temporary placeholder

        if len(hyperparams) != len(hyperparams_fitted):
            logging.error(
                f'Ignoring following kwargs for model constructor: '
                f'{set(hyperparams) - set(hyperparams_fitted)}.'
            )

        return constructor(**hyperparams_fitted)
        
    @staticmethod
    def fit_model(model, X_train, y_train, s_train = None):
        sig = signature(model.fit)

        # compatibility with fairgbm
        if 'constraint_group' in sig.parameters:
            model.fit(X_train, y_train, constraint_group=s_train)

        # compatibility with fairlearn
        elif 'sensitive_features' in sig.parameters:
            model.fit(X_train, y_train, sensitive_features=s_train)
        
        # TODO: add adhoc compatibility with other libraries here

        # else, attempt to use s_train as a third positional argument
        elif s_train is not None and len(sig.parameters) > 2:
            model.fit(X_train, y_train, s_train)

        # else, train without sensitive attribute data
        else:
            if s_train is not None:
                logging.error(
                    f"Can't figure out how to use sensitive_attribute data for "
                    f"training with object of type '{type(model)}'.")

            model.fit(X_train, y_train)

        return model
    
    @property
    def results(self):
        return pd.DataFrame(
            data=[{
                'algorithm': r.algorithm,
                **r.validation_results,
            } for r in self._models_results],
            index=[r.id for r in self._models_results],
        )
    
    @property
    def all_results(self):
        return self._models_results
    
    @property
    def best_trial(self):
        results = self.results.copy()
        target_metric_col = self.eval_metric

        if self.other_eval_metric:
            target_metric_col = 'weighted_metric'
            results[target_metric_col] = (
                results[self.eval_metric] * self.alpha + 
                results[self.other_eval_metric] * (1-self.alpha))
        
        # NOTE: trial_idx != trial.id
        best_trial_idx = np.argmax(results[target_metric_col])
        return self.all_results[best_trial_idx]

    # NOTE: I don't love this constructor API, feels cluttered
    def __init__(
            self,
            X_train,
            y_train,
            X_val,
            y_val,
            hyperparameter_space: Union[str, dict],
            eval_metric: str,
            s_train = None,
            s_val = None,
            other_eval_metric: Optional[str] = None,
            alpha: Optional[float] = 0.50,
            eval_func: Optional[Callable[..., dict]] = None,
            **threshold_target,
        ):
        self.X_train, self.y_train, self.s_train = (X_train, y_train, s_train)
        self.X_val, self.y_val, self.s_val = (X_val, y_val, s_val)

        self.hyperparameter_space = hyperparameter_space
        if isinstance(hyperparameter_space, (str, Path)):
            self.hyperparameter_space = load_hyperparameter_space(str(hyperparameter_space))

        self.eval_metric = eval_metric
        self.other_eval_metric = other_eval_metric
        self.alpha = alpha
        assert alpha is None or (0 <= alpha <= 1)
        self.eval_func = eval_func or evaluate_predictions
        self.eval_func = partial(self.eval_func, **threshold_target)

        # Store all results in a list as models are trained
        self._models_results: List[ObjectiveFunction.TrialResults] = list()

    def __call__(self, trial: BaseTrial) -> float:

        # Sample hyperparameters
        hyperparams = suggest_callable_hyperparams(trial, self.hyperparameter_space)

        # Construct model
        model = self.instantiate_model(**hyperparams)

        # Train model
        self.fit_model(model, self.X_train, self.y_train, self.s_train)

        # Compute predictions on validation data
        y_val_pred_scores = model.predict_proba(self.X_val)
        # TODO: computing predictions may require sensitive attribute access

        if len(y_val_pred_scores.shape) > 1:
            y_val_pred_scores = y_val_pred_scores[:, -1]

        # Evaluate validation predictions
        val_results = self.eval_func(
            self.y_val, y_val_pred_scores,
            sensitive_attribute=self.s_val,
        )

        # Store trial's results
        self._models_results.append(
            self.TrialResults(
                id=trial.number,
                hyperparameters=hyperparams,
                validation_results=val_results,
                model=model
            ))

        # Return scalarized evaluation metric
        if self.other_eval_metric:
            assert self.alpha is not None
            return (
                val_results[self.eval_metric] * self.alpha + 
                val_results[self.other_eval_metric] * (1 - self.alpha)
            )
        
        # Or simply a single evaluation metric value
        else:
            return val_results[self.eval_metric]
    
    def reconstruct_model(self, trial_results: TrialResults):
        model = self.instantiate_model(**trial_results.hyperparameters)
        return self.fit_model(model, self.X_train, self.y_train, self.s_train)
    
    def plot(
            self,
            x_axis: str = None,
            y_axis: str = None,
            pyplot_show: bool = True,
            **kwargs,
        ):
        x_axis = x_axis or self.eval_metric
        y_axis = y_axis or self.other_eval_metric
        if y_axis is None:
            raise TypeError(
                "No y_axis metric could be inferred as only one optimization "
                "metric was used. Please provide kwarg `y_axis`."
            )

        try:
            import seaborn as sns
            from matplotlib import pyplot as plt

        except ImportError as err:
            logging.error(
                f"Necessary dependencies for plotting were not found: {err}."
                f"You can install them with `pip install hpt[plotting]`."
            )

        sns.set()
        sns.scatterplot(self.results, x=x_axis, y=y_axis, **kwargs)

        plt.title("Hyperparameter Search")

        if pyplot_show:
            plt.show()

        # TODO: plot the best model with a star
        # sns.scatterplot(
        #     x=[results.loc[best_trial_id][PERFORMANCE_METRIC]], y=[results.loc[best_trial_id]['equal_odds_diff']],
        #     color='red', marker='*', s=100,
        #     label='best model',
        # )


class OptunaTuner:
    """This class is mostly useless, just a helper for common boilerplate.
    """

    def __init__(
            self,
            objective_function: Callable[[BaseTrial], float],
            sampler: Optional[samplers.BaseSampler] = None,
            seed: int = 42,
            **study_kwargs,
        ):

        self.objective_function = objective_function
        self.seed = seed
        self.study = optuna.create_study(
            sampler=(sampler or samplers.RandomSampler(self.seed)),
            **study_kwargs,
        )

    @property
    def results(self):
        return self.objective_function.results
    
    def optimize(self, **kwargs):
        return self.study.optimize(
            self.objective_function,
            **kwargs,
        )
