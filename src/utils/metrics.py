import numpy as np
from typing import Union

Number = Union[int, float]

def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Mean Absolute Error (MAE) between actual and predicted values.
    """
    return float(np.mean(np.abs(actual - predicted)))

def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error (RMSE) between actual and predicted values.
    """
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))