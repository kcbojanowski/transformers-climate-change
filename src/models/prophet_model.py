import pandas as pd
from prophet import Prophet
from typing import Any


def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on the given DataFrame.
    The DataFrame must contain columns 'ds' (dates) and 'y' (target variable).

    Parameters:
        df (pd.DataFrame): The DataFrame with columns 'ds' and 'y'.

    Returns:
        Prophet: The trained Prophet model.
    """
    model = Prophet()
    model.fit(df)
    return model


def forecast_prophet(model: Prophet, periods: int = 10) -> pd.DataFrame:
    """
    Forecast future values using the trained Prophet model.

    Parameters:
        model (Prophet): The trained Prophet model.
        periods (int): Number of periods (days) to forecast.

    Returns:
        pd.DataFrame: A DataFrame containing the forecasted values.
    """
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]