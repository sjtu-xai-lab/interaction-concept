import os

# import ML libs
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# import internal libs
from .train_classification import train_classification_model, eval_classification_model
from .train_regression import train_regression_model, eval_regression_model
from .train_logistic_regression import train_logistic_regression_model, eval_logistic_regression_model
from .train_pointcloud_classification import train_pointcloud_classification_model, eval_pointcloud_classification_model


def train_model(args, model, train_loader, test_loader, task="classification"):
    if task == "classification":
        return train_classification_model(args, model, train_loader, test_loader)
    elif task == "regression":
        return train_regression_model(args, model, train_loader, test_loader)
    elif task == "logistic_regression":
        return train_logistic_regression_model(args, model, train_loader, test_loader)
    elif task == "pointcloud_classification":
        return train_pointcloud_classification_model(args, model, train_loader, test_loader)
    else:
        raise NotImplementedError(f"Unknown task: {task}.")


def eval_model(model, data_loader, task="classification"):
    if task == "classification":
        return eval_classification_model(model=model, test_loader=data_loader, criterion=nn.CrossEntropyLoss())
    elif task == "regression":
        return eval_regression_model(model=model, test_loader=data_loader, criterion=nn.MSELoss())
    elif task == "logistic_regression":
        return eval_logistic_regression_model(model=model, test_loader=data_loader, criterion=nn.BCEWithLogitsLoss())
    elif task == "pointcloud_classification":
        return eval_pointcloud_classification_model(model=model, test_loader=data_loader, criterion=nn.CrossEntropyLoss())
    else:
        raise NotImplementedError(f"Unknown task: {task}.")