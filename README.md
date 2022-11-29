# hpt

> This repository is under construction :construction:

![badge for tests status](https://github.com/AndreFCruz/hpt/actions/workflows/python-package.yml/badge.svg)
![badge for PyPI publishing status](https://github.com/AndreFCruz/hpt/actions/workflows/python-publish.yml/badge.svg)

A minimal hyperparameter tuning framework to help you train hundreds of models.

It's essentially a set of helpful wrappers over optuna.


## Install

`
pip install hpt
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
```

**_TODO:_** finish readme.


<!-- ## Defining a hyperparameter space -->
