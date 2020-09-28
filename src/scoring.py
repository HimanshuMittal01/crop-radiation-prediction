import numpy as np
from functools import wraps
from sklearn.metrics import mean_squared_error

def _root(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        return np.sqrt(func(*args, **kwargs))
    
    return wrapped

class SklearnScorer:
    """Namespace which have scoring methods for sklearn models
    Returns f(y_true, y_pred)
    """

    @staticmethod
    def get_scoring_fn(scoring):
        if scoring=="mse" or scoring=="mean_squared_error":
            return mean_squared_error
        elif scoring=="rmse" or scoring=="root_mean_squared_error":
            return _root(mean_squared_error)
    