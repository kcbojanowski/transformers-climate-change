import os
import time
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import r2_score, explained_variance_score
from darts import TimeSeries

from src.config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_SAVE_DIR,
    ARIMA_ORDER,
    TRANSFORMER_CONFIG,
    DARTS_TRANSFORMER_CONFIG,
    TARGET_VARIABLES,
)

# Import models
from src.models.arima_model import train_arima_model, forecast_arima
from src.models.prophet_model import train_prophet_model, forecast_prophet
from src.models.transformer_model import TransformerModel, DartsTransformerWrapper
from src.utils.result_utils import convert_np_types
from src.utils.monitoring import log_system_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def evaluate_regression(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    r2 = r2_score(actual, predicted)
    explained_var = explained_variance_score(actual, predicted)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2,
        "ExplainedVariance": explained_var
    }


def load_data():
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, val_df, test_df


def run_arima_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting ARIMA experiment...")
    start = time.time()
    train_series = train_df.groupby('time_idx')['t2m'].mean()
    model = train_arima_model(train_series, order=ARIMA_ORDER)
    forecast_steps = len(val_df['time_idx'].unique())
    forecast = forecast_arima(model, steps=forecast_steps)
    elapsed = time.time() - start
    results = {"training_time": elapsed, "forecast_steps": forecast_steps}
    val_series = val_df.groupby('time_idx')['t2m'].mean().reindex(forecast.index)
    metrics = evaluate_regression(val_series.values, forecast.values)
    results.update(metrics)
    results["system_metrics"] = log_system_metrics()
    logging.info(f"ARIMA results: {results}")
    with open(os.path.join(MODEL_SAVE_DIR, "arima_model_summary.txt"), "w") as f:
        f.write(str(model.summary()))
    # Also save the forecast series for plotting
    results["forecast_series"] = forecast.tolist()
    return results


def run_prophet_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting Prophet experiment...")
    start = time.time()
    prophet_train = train_df.groupby('time_idx').agg({"time": "first", "t2m": "mean"}).reset_index()
    prophet_train.rename(columns={'time': 'ds', 't2m': 'y'}, inplace=True)
    model = train_prophet_model(prophet_train)
    forecast_df = forecast_prophet(model, periods=len(val_df['time_idx'].unique()))
    elapsed = time.time() - start
    results = {"training_time": elapsed}
    prophet_val = val_df.groupby('time_idx').agg({"time": "first", "t2m": "mean"}).reset_index()
    prophet_val.rename(columns={'time': 'ds', 't2m': 'y'}, inplace=True)
    forecast_yhat = forecast_df['yhat'].tail(len(prophet_val)).values
    metrics = evaluate_regression(prophet_val['y'].values, forecast_yhat)
    results.update(metrics)
    results["system_metrics"] = log_system_metrics()
    logging.info(f"Prophet results: {results}")
    # Save Prophet model via pickle
    import pickle
    with open(os.path.join(MODEL_SAVE_DIR, "prophet_model.pkl"), "wb") as f:
        pickle.dump(model, f)
    results["forecast_series"] = forecast_df['yhat'].tolist()
    return results


def run_transformer_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting Custom Transformer experiment...")
    start = time.time()
    sequence_length = 30

    grouped_train = train_df.groupby('time_idx')[TARGET_VARIABLES].mean()
    series = grouped_train.values  # shape: (num_days, num_vars)

    X, y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i + sequence_length])
        y.append(series[i + sequence_length])
    X = np.array(X)
    y = np.array(y)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    train_loader = [(X_tensor, y_tensor)]
    val_loader = [(X_tensor, y_tensor)]

    model_params = {
        "input_size": TRANSFORMER_CONFIG["input_size"],
        "model_dim": TRANSFORMER_CONFIG["model_dim"],
        "num_heads": TRANSFORMER_CONFIG["num_heads"],
        "num_layers": TRANSFORMER_CONFIG["num_layers"],
        "output_size": TRANSFORMER_CONFIG["output_size"],
        "dropout": TRANSFORMER_CONFIG["dropout"],
    }
    num_epochs = TRANSFORMER_CONFIG["num_epochs"]
    lr = TRANSFORMER_CONFIG["lr"]
    transformer_model = TransformerModel(**model_params).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    train_losses, val_losses = [], []

    for epoch in tqdm(range(num_epochs), desc="Custom Transformer Training"):
        transformer_model.train()
        epoch_train_loss = 0.0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = transformer_model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss)
        transformer_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = transformer_model(x_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss)
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    elapsed = time.time() - start
    results = {"training_time": elapsed, "train_loss": train_losses, "val_loss": val_losses}
    transformer_model.eval()
    with torch.no_grad():
        transformer_forecast = transformer_model(X_tensor[-1:]).detach().cpu().numpy().squeeze().tolist()
    target = y_tensor[-1].detach().cpu().numpy().squeeze().tolist()
    metrics = evaluate_regression(np.array(target), np.array(transformer_forecast))
    results.update(metrics)
    results["system_metrics"] = log_system_metrics()
    logging.info(f"Custom Transformer results: {results}")
    torch.save(transformer_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "transformer_model.pth"))
    results["forecast_series"] = transformer_forecast
    return results


def run_darts_transformer_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting Darts Transformer experiment...")
    # Remove duplicate dates if needed
    train_df_unique = train_df.drop_duplicates(subset=['time'])
    val_df_unique = val_df.drop_duplicates(subset=['time'])

    # Convert train_df to a Darts TimeSeries with daily frequency
    series = TimeSeries.from_dataframe(
        train_df_unique,
        time_col='time',
        value_cols=['t2m'],
        fill_missing_dates=True,
        freq='D'
    )

    darts_start = time.time()
    darts_model = DartsTransformerWrapper(DARTS_TRANSFORMER_CONFIG)
    darts_model.fit(series)
    darts_elapsed = time.time() - darts_start
    forecast = darts_model.predict(n=len(val_df_unique['time_idx'].unique()))
    forecast_df = forecast.pd_dataframe()
    val_series = TimeSeries.from_dataframe(
        val_df_unique,
        time_col='time',
        value_cols=['t2m'],
        fill_missing_dates=True,
        freq='D'
    )
    metrics = evaluate_regression(val_series.values().squeeze(), forecast_df['t2m'].values)
    results = {
        "training_time": darts_elapsed,
        "metrics": metrics,
        "forecast_series": forecast_df.to_dict(orient="list")
    }
    results["system_metrics"] = log_system_metrics()
    return results

def run_all_experiments():
    train_df, val_df, test_df = load_data()
    results = {}
    results["ARIMA"] = run_arima_experiment(train_df, val_df)
    results["Prophet"] = run_prophet_experiment(train_df, val_df)
    results["CustomTransformer"] = run_transformer_experiment(train_df, val_df)
    #results["DartsTransformer"] = run_darts_transformer_experiment(train_df, val_df)
    results["global_system_metrics"] = log_system_metrics()
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["hyperparameters"] = {
        "ARIMA": {"order": ARIMA_ORDER},
        "Prophet": {},
        "CustomTransformer": TRANSFORMER_CONFIG,
        #"DartsTransformer": DARTS_TRANSFORMER_CONFIG,
    }
    results_file = os.path.join(MODEL_SAVE_DIR, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(convert_np_types(results), f, indent=4)
    logging.info(f"Experiment results saved to {results_file}")
    print("\nSummary of Experiment Results:")
    print(json.dumps(convert_np_types(results), indent=4))

if __name__ == "__main__":
    run_all_experiments()