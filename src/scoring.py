import numpy as np
import torch
from functools import wraps
from sklearn import metrics

def _np_root(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return np.sqrt(func(*args, **kwargs))
    
    return wrapped

def _torch_root(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return torch.sqrt(func(*args, **kwargs))
    
    return wrapped

class SklearnScorer:
    """Namespace which have scoring methods for sklearn models
    Returns f(y_true, y_pred)
    """

    @staticmethod
    def get_scoring_fn(scoring):
        if scoring=="mse" or scoring=="mean_squared_error":
            return metrics.mean_squared_error
        elif scoring=="rmse" or scoring=="root_mean_squared_error":
            return _np_root(metrics.mean_squared_error)

class TorchScorer:
    """Namespace which have scoring methods for torch models
    Returns f(y_true, y_pred)
    """

    @staticmethod
    def get_scoring_fn(scoring):
        if scoring=="mse" or scoring=="mean_squared_error":
            return torch.nn.MSELoss()
        elif scoring=="rmse" or scoring=="root_mean_squared_error":
            return _torch_root(torch.nn.MSELoss())
    