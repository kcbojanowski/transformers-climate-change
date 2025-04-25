import numpy as np
from sklearn.metrics import r2_score, explained_variance_score


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray) -> dict:
    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted) if len(actual) > 1 else float('nan')
    explained_var = explained_variance_score(actual, predicted) if len(actual) > 1 else float('nan')

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "ExplainedVariance": explained_var
    }