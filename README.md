# hpt

> This repository is under construction :construction:

![badge for tests status](https://github.com/AndreFCruz/hpt/actions/workflows/python-package.yml/badge.svg)
![badge for PyPI publishing status](https://github.com/AndreFCruz/hpt/actions/workflows/python-publish.yml/badge.svg)

A minimal hyperparameter tuning framework to help you train hundreds of models.

It's essentially a set of helpful wrappers over optuna.


## Install

Install package from [PyPI](https://pypi.org/project/hyperparameter-tuning/):


`
pip install hyperparameter-tuning
`

## Getting started

```py
from hpt.tuner import ObjectiveFunction, OptunaTuner

obj_func = ObjectiveFunction(
    X_train, y_train, X_test, y_test,
    hyperparameter_space=HYPERPARAM_SPACE_PATH,
    eval_metric='accuracy',
    s_train=s_train,
    s_val=s_test,
    threshold=0.50,
)

tuner = OptunaTuner(obj_func) # NOTE: can pass other useful study kwargs here (e.g. storage)

# Then just run optimize as you would for an optuna.Study object
tuner.optimize(n_trials=20, n_jobs=4)

# Results are stored in tuner.results
tuner.results
```

## Defining a hyperparameter space

The hyperparameter space is provided either path to a YAML file, or as a `dict` 
with the same structure.
Example hyperparameter spaces [here](examples/sklearn.small_hyperparam_space.yaml) and 
[here](examples/sklearn.large_hyperparam_space.yaml).

The YAML file must follow this structure:
```yaml
# One or more top-level algorithms
DT:  
    # Full classpath of algorithm's constructor
    classpath: sklearn.tree.DecisionTreeClassifier
    
    # One or more key-word arguments to be passed to the constructor
    kwargs:
        
        # Kwargs may be sampled from a distribution
        max_depth:
            type: int           # either 'int' or 'float'
            range: [ 10, 100 ]  # minimum and maximum values
            log: True           # (optionally) whether to use logarithmic scale
        
        # Kwargs may be sampled from a fixed set of categories
        criterion:
            - 'gini'
            - 'entropy'
        
        # Kwargs may be a pre-defined value
        min_samples_split: 4


# You may explore multiple algorithms at once
LR:
    classpath: sklearn.linear_model.LogisticRegression
    kwargs:
        # An example of a float hyperparameter
        C:
            type: float
            range: [ 0.01, 1.0 ]
            log: True

```
