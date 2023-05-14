import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from .train_classification_adv import train_classification_model_adv, eval_classification_model, \
    eval_classification_model_adv
from .train_regression_adv import train_regression_model_adv, eval_regression_model, eval_regression_model_adv
from .train_logistic_regression_adv import train_logistic_regression_model_adv, eval_logistic_regression_model, \
    eval_logistic_regression_model_adv


def train_model_adv(
        args, model, train_loader, test_loader,
        adv_step_size, adv_epsilon, adv_n_step,
        adv_bound_max=None, adv_bound_min=None,
        normalize_fn=None, denormalize_fn=None,
        task="classification"
):
    if task == "classification":
        return train_classification_model_adv(
            args, model, train_loader, test_loader,
            adv_step_size, adv_epsilon, adv_n_step,
            adv_bound_max, adv_bound_min,
            normalize_fn, denormalize_fn
        )
    elif task == "regression":
        return train_regression_model_adv(
            args, model, train_loader, test_loader,
            adv_step_size, adv_epsilon, adv_n_step,
            adv_bound_max, adv_bound_min,
            normalize_fn, denormalize_fn
        )
    elif task == "logistic_regression":
        return train_logistic_regression_model_adv(
            args, model, train_loader, test_loader,
            adv_step_size, adv_epsilon, adv_n_step,
            adv_bound_max, adv_bound_min,
            normalize_fn, denormalize_fn
        )
    else:
        raise NotImplementedError(f"Unknown task: {task}.")


def eval_model(model, data_loader, task="classification"):
    if task == "classification":
        return eval_classification_model(model=model, test_loader=data_loader, criterion=nn.CrossEntropyLoss())
    elif task == "regression":
        return eval_regression_model(model=model, test_loader=data_loader, criterion=nn.MSELoss())
    elif task == "logistic_regression":
        return eval_logistic_regression_model(model=model, test_loader=data_loader, criterion=nn.BCEWithLogitsLoss())
    else:
        raise NotImplementedError(f"Unknown task: {task}.")


def eval_model_adv(
        model, data_loader,
        adv_step_size, adv_epsilon, adv_n_step,
        adv_bound_max=None, adv_bound_min=None,
        normalize_fn=None, denormalize_fn=None,
        task="classification"
):
    if task == "classification":
        return eval_classification_model_adv(
            model=model, test_loader=data_loader, criterion=nn.CrossEntropyLoss(),
            adv_step_size=adv_step_size, adv_epsilon=adv_epsilon, adv_n_step=adv_n_step,
            adv_bound_max=adv_bound_max, adv_bound_min=adv_bound_min,
            normalize_fn=normalize_fn, denormalize_fn=denormalize_fn
        )
    elif task == "regression":
        return eval_regression_model_adv(
            model=model, test_loader=data_loader, criterion=nn.MSELoss(),
            adv_step_size=adv_step_size, adv_epsilon=adv_epsilon, adv_n_step=adv_n_step,
            adv_bound_max=adv_bound_max, adv_bound_min=adv_bound_min,
            normalize_fn=normalize_fn, denormalize_fn=denormalize_fn
        )
    elif task == "logistic_regression":
        return eval_logistic_regression_model_adv(
            model=model, test_loader=data_loader, criterion=nn.BCEWithLogitsLoss(),
            adv_step_size=adv_step_size, adv_epsilon=adv_epsilon, adv_n_step=adv_n_step,
            adv_bound_max=adv_bound_max, adv_bound_min=adv_bound_min,
            normalize_fn=normalize_fn, denormalize_fn=denormalize_fn
        )
    else:
        raise NotImplementedError(f"Unknown task: {task}.")