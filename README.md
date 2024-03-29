# hpt

![Tests status](https://github.com/AndreFCruz/hpt/actions/workflows/python-package.yml/badge.svg)
![PyPI status](https://github.com/AndreFCruz/hpt/actions/workflows/python-publish.yml/badge.svg)
![Documentation status](https://github.com/AndreFCruz/hpt/actions/workflows/python-docs.yml/badge.svg)
![PyPI version](https://badgen.net/pypi/v/hyperparameter-tuning)
![OSI license](https://badgen.net/pypi/license/hyperparameter-tuning)
![Python compatibility](https://badgen.net/pypi/python/hyperparameter-tuning)

A minimal hyperparameter tuning framework to help you train hundreds of models.

It's essentially a set of helpful wrappers over optuna.

Consult the package documentation [here](https://andrefcruz.github.io/hpt/)!


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
    hyperparameter_space=HYPERPARAM_SPACE_PATH,    # path to YAML file
    eval_metric="accuracy",
    s_train=s_train,
    s_val=s_test,
    threshold=0.50,
)

tuner = OptunaTuner(
    objective_function=obj_func,
    direction="maximize",    # NOTE: can pass other useful study kwargs here (e.g. storage)
)

# Then just run optimize as you would for an optuna.Study object
tuner.optimize(n_trials=20, n_jobs=4)

# Results are stored in tuner.results
tuner.results

# You can reconstruct the best predictor with:
clf = obj_func.reconstruct_model(obj_func.best_trial)
```

## Defining a hyperparameter space

The hyperparameter space is provided either path to a YAML file, or as a `dict` 
with the same structure.
Example hyperparameter spaces [here](examples/hyperparameter_spaces/).

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
