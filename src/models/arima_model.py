import pandas as pd
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from typing import Tuple


def train_arima_model(series: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> ARIMAResults:
    """
    Train an ARIMA model on a given time series.

    Parameters:
        series (pd.Series): The time series data.
        order (tuple): The (p, d, q) order of the ARIMA model.

    Returns:
        ARIMAResults: The fitted ARIMA model.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    return model_fit


def forecast_arima(model_fit: ARIMAResults, steps: int = 10) -> pd.Series:
    """
    Forecast future values using the trained ARIMA model.

    Parameters:
        model_fit (ARIMAResults): The fitted ARIMA model.
        steps (int): Number of time steps to forecast.

    Returns:
        pd.Series: The forecasted values.
    """
    forecast = model_fit.forecast(steps=steps)
    return forecast