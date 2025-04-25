import os
import time
import json
import logging
import pickle
from typing import Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from src.models.arima_model import train_arima_model, forecast_arima
from src.models.prophet_model import train_prophet_model, forecast_prophet
from src.models.transformer_model import TransformerModel, DartsTransformerWrapper
from darts import TimeSeries

from src.utils.result_utils import convert_np_types
from src.utils.monitoring import log_system_metrics
from src.utils.metrics import evaluate_regression

import src.config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def load_data():
    train_df = pd.read_csv(src.config.TRAIN_DATA_PATH)
    val_df = pd.read_csv(src.config.VAL_DATA_PATH)
    test_df = pd.read_csv(src.config.TEST_DATA_PATH)
    return train_df, val_df, test_df


def run_arima_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting ARIMA experiment...")
    start = time.time()

    train_series = train_df.groupby('time_idx')['t2m'].mean()
    model = train_arima_model(train_series, order=src.config.ARIMA_ORDER)
    forecast_steps = len(val_df['time_idx'].unique())
    forecast = forecast_arima(model, steps=forecast_steps)

    elapsed = time.time() - start
    results = {"training_time": elapsed, "forecast_steps": forecast_steps}

    val_series = val_df.groupby('time_idx')['t2m'].mean().reindex(forecast.index)
    metrics = evaluate_regression(val_series.values, forecast.values)
    results.update(metrics)
    results["system_metrics"] = log_system_metrics()

    logging.info(f"ARIMA results: {results}")
    with open(os.path.join(src.config.MODEL_SAVE_DIR, "arima_model_summary.txt"), "w") as f:
        f.write(str(model.summary()))

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
    with open(os.path.join(src.config.MODEL_SAVE_DIR, "prophet_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    results["forecast_series"] = forecast_df['yhat'].tolist()
    return results


def run_transformer_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Dict:
    logging.info("Starting Custom Transformer experiment...")
    start = time.time()

    sequence_length = src.config.SEQUENCE_LENGTH
    grouped_train = train_df.groupby('time_idx')[src.config.TARGET_VARIABLES].mean()
    series = grouped_train.values

    X, y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i + sequence_length])
        y.append(series[i + sequence_length])
    X = np.array(X)
    y = np.array(y)

    # Logging number of training samples
    logging.info(f"Number of training samples: {len(X)}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    train_loader = [(X_tensor, y_tensor)]
    val_loader = [(X_tensor, y_tensor)]

    model_params = {
        "input_size": src.config.TRANSFORMER_CONFIG["input_size"],
        "model_dim": src.config.TRANSFORMER_CONFIG["model_dim"],
        "num_heads": src.config.TRANSFORMER_CONFIG["num_heads"],
        "num_layers": src.config.TRANSFORMER_CONFIG["num_layers"],
        "output_size": src.config.TRANSFORMER_CONFIG["output_size"],
        "dropout": src.config.TRANSFORMER_CONFIG["dropout"],
    }
    num_epochs = src.config.TRANSFORMER_CONFIG["num_epochs"]
    lr = src.config.TRANSFORMER_CONFIG["lr"]
    transformer_model = TransformerModel(**model_params).to(device)
    # Log model structure
    logging.info(f"Transformer Model Structure:\n{transformer_model}")
    optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_losses, val_losses, system_metrics = [], [], []
    for epoch in tqdm(range(num_epochs), desc="Custom Transformer Training"):
        # Log current epoch
        logging.info(f"Epoch {epoch + 1}/{num_epochs} started.")
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
        current_system_metrics = log_system_metrics()
        system_metrics.append(current_system_metrics)

        transformer_model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                outputs = transformer_model(x_batch)
                loss = criterion(outputs, y_batch)
                epoch_val_loss += loss.item()
        val_losses.append(epoch_val_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

    elapsed = time.time() - start
    results = {
        "training_time": elapsed,
        "train_loss": train_losses,
        "val_loss": val_losses,
        "system_metrics": system_metrics
    }

    forecast_horizon = src.config.FORECAST_HORIZON

    transformer_model.eval()
    forecast_series = []
    input_seq = X_tensor[-1:].clone()  # Start with last known sequence

    with torch.no_grad():
        for _ in range(forecast_horizon):
            next_step = transformer_model(input_seq)  # Shape: (1, output_size)
            forecast_series.append(next_step.squeeze().cpu().numpy())

            # Append prediction to the sequence and slide window
            next_step_expanded = next_step.unsqueeze(1)  # Shape: (1, 1, output_size)
            input_seq = torch.cat((input_seq[:, 1:, :], next_step_expanded), dim=1)

    forecast_series = np.array(forecast_series)

    # Evaluate just on the first predicted step (optional)
    target = y_tensor[-1].detach().cpu().numpy().flatten()
    metrics = evaluate_regression(np.array([target[0]]), np.array([forecast_series[0]]))
    results.update(metrics)

    results["forecast_series"] = forecast_series.tolist()

    torch.save(transformer_model.state_dict(), os.path.join(src.config.MODEL_SAVE_DIR, "transformer_model.pth"))
    logging.info("Transformer model saved successfully.")

    return results



def run_darts_transformer_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    logging.info("Starting Darts Transformer experiment...")

    train_df_unique = train_df.drop_duplicates(subset=['time'])
    val_df_unique = val_df.drop_duplicates(subset=['time'])

    series = TimeSeries.from_dataframe(
        train_df_unique,
        time_col='time',
        value_cols=['t2m'],
        fill_missing_dates=True,
        freq='D'
    )

    darts_start = time.time()
    darts_model = DartsTransformerWrapper(src.config.DARTS_TRANSFORMER_CONFIG)
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
        "forecast_series": forecast_df.to_dict(orient="list"),
        "system_metrics": log_system_metrics()
    }
    return results


def evaluate_horizons_arima(train_df: pd.DataFrame, test_df: pd.DataFrame, max_horizon: int = src.config.FORECAST_HORIZON,
                            threshold: float = 1.0) -> dict:
    logging.info("Ocena horyzontów dla modelu ARIMA rozpoczęta.")
    train_series = train_df.groupby('time_idx')['t2m'].mean()
    model = train_arima_model(train_series, order=src.config.ARIMA_ORDER)
    horizon_errors = {}

    logging.info("Przechodzenie przez horyzonty prognoz (ARIMA)...")
    for h in tqdm(range(1, max_horizon + 1), desc="Horyzonty ARIMA"):
        forecast = forecast_arima(model, steps=h)
        test_series = test_df.groupby('time_idx')['t2m'].mean().head(h)

        if len(forecast) != len(test_series):
            print(
                f"Warning: dla horyzontu {h} forecast length {len(forecast)} nie odpowiada test length {len(test_series)}")

        min_len = min(len(test_series), len(forecast))
        metrics = evaluate_regression(test_series.values[:min_len], forecast.values[:min_len])
        horizon_errors[h] = metrics["RMSE"]
        logging.info(f"Horyzont {h}: RMSE = {horizon_errors[h]:.4f}")

    feasible = [h for h, err in horizon_errors.items() if err < threshold]
    max_feasible = max(feasible) if feasible else 0
    logging.info("Ocena horyzontów dla ARIMA zakończona.")

    return {"horizon_errors": horizon_errors, "max_feasible_horizon": max_feasible}


def evaluate_horizons_transformer(train_df: pd.DataFrame, test_df: pd.DataFrame, sequence_length: int = src.config.SEQUENCE_LENGTH,
                                  max_horizon: int = src.config.FORECAST_HORIZON, threshold: float = 1.0) -> dict:
    logging.info("Ocena horyzontów dla Custom Transformer rozpoczęta.")
    grouped_train = train_df.groupby('time_idx')['t2m'].mean()
    train_series = grouped_train.values  # shape: (num_days,)
    model_params = {
        "input_size": src.config.TRANSFORMER_CONFIG["input_size"],
        "model_dim": src.config.TRANSFORMER_CONFIG["model_dim"],
        "num_heads": src.config.TRANSFORMER_CONFIG["num_heads"],
        "num_layers": src.config.TRANSFORMER_CONFIG["num_layers"],
        "output_size": src.config.TRANSFORMER_CONFIG["output_size"],
        "dropout": src.config.TRANSFORMER_CONFIG["dropout"],
    }
    num_epochs = src.config.TRANSFORMER_CONFIG["num_epochs"]
    lr = src.config.TRANSFORMER_CONFIG["lr"]
    transformer_model = TransformerModel(**model_params).to(device)

    X, y = [], []
    for i in range(len(train_series) - sequence_length):
        X.append(train_series[i:i + sequence_length])
        y.append(train_series[i + sequence_length])

    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    logging.info("Trening Custom Transformer...")
    for epoch in tqdm(range(num_epochs), desc="Trening Transformer"):
        transformer_model.train()
        optimizer.zero_grad()
        outputs = transformer_model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

    logging.info("Trening zakończony. Ocena horyzontów dla Custom Transformer...")
    grouped_test = test_df.groupby('time_idx')['t2m'].mean()
    test_series = grouped_test.values  # shape: (N_test,)
    forecast_errors = {}

    for h in tqdm(range(1, max_horizon + 1), desc="Horyzonty Transformer"):
        current_input = test_series[-sequence_length:].copy()
        preds = []
        for _ in range(h):
            input_tensor = torch.tensor(current_input, dtype=torch.float32).reshape(1, sequence_length, 1).to(device)
            with torch.no_grad():
                pred = transformer_model(input_tensor).detach().cpu().numpy().squeeze()
            preds.append(pred)
            current_input = np.append(current_input[1:], pred)
        preds = np.array(preds)
        actual = test_series[-h:]
        min_len = min(len(actual), len(preds))
        metrics = evaluate_regression(actual[:min_len], preds[:min_len])
        forecast_errors[h] = metrics["RMSE"]
        logging.info(f"Horyzont {h}: RMSE = {forecast_errors[h]:.4f}")

    feasible = [h for h, err in forecast_errors.items() if err < threshold]
    max_feasible = max(feasible) if feasible else 0
    logging.info("Ocena horyzontów dla Custom Transformer zakończona.")
    return {"horizon_errors": forecast_errors, "max_feasible_horizon": max_feasible}


def evaluate_horizons_prophet(train_df: pd.DataFrame, test_df: pd.DataFrame, max_horizon: int = src.config.FORECAST_HORIZON,
                              threshold: float = 1.0) -> dict:
    logging.info("Ocena horyzontów dla Prophet rozpoczęta.")
    prophet_train = train_df.groupby('time_idx').agg({"time": "first", "t2m": "mean"}).reset_index()
    prophet_train.rename(columns={'time': 'ds', 't2m': 'y'}, inplace=True)

    model = train_prophet_model(prophet_train)
    horizon_errors = {}
    grouped_test = test_df.groupby('time_idx')['t2m'].mean()

    for h in tqdm(range(1, max_horizon + 1), desc="Horyzonty Prophet"):
        future = model.make_future_dataframe(periods=h)
        forecast_df = model.predict(future)
        forecast_h = forecast_df.tail(h)
        test_h = grouped_test.iloc[:h]

        if len(forecast_h) != len(test_h):
            print(
                f"Warning: dla horyzontu {h} długość forecast {len(forecast_h)} nie odpowiada długości test {len(test_h)}")

        min_len = min(len(test_h), len(forecast_h))
        metrics = evaluate_regression(test_h.values[:min_len], forecast_h['yhat'].values[:min_len])
        horizon_errors[h] = metrics["RMSE"]
        logging.info(f"Horyzont {h}: RMSE = {horizon_errors[h]:.4f}")

    feasible = [h for h, err in horizon_errors.items() if err < threshold]
    max_feasible = max(feasible) if feasible else 0
    logging.info("Ocena horyzontów dla Prophet zakończona.")
    return {"horizon_errors": horizon_errors, "max_feasible_horizon": max_feasible}


def run_horizon_experiments():
    logging.info("Uruchamianie eksperymentów horyzontowych...")
    train_df, val_df, test_df = load_data()
    horizon_results = {}
    RMSE_threshold = 1.0

    logging.info("Ocena horyzontów dla ARIMA...")
    arima_horizons = evaluate_horizons_arima(train_df, test_df, threshold=RMSE_threshold)
    horizon_results["ARIMA"] = arima_horizons

    logging.info("Ocena horyzontów dla Prophet...")
    prophet_horizons = evaluate_horizons_prophet(train_df, test_df, threshold=RMSE_threshold)
    horizon_results["Prophet"] = prophet_horizons

    logging.info("Ocena horyzontów dla Custom Transformer...")
    transformer_horizons = evaluate_horizons_transformer(train_df, test_df, threshold=RMSE_threshold)
    horizon_results["CustomTransformer"] = transformer_horizons

    horizon_results["global_system_metrics"] = log_system_metrics()
    horizon_results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    horizon_results["hyperparameters"] = {
        "ARIMA": {"order": src.config.ARIMA_ORDER},
        "Prophet": {},
        "CustomTransformer": src.config.TRANSFORMER_CONFIG,
    }

    horizon_results_file = os.path.join(src.config.MODEL_SAVE_DIR, "test_horizon_results.json")
    with open(horizon_results_file, "w") as f:
        json.dump(convert_np_types(horizon_results), f, indent=4)
    logging.info(f"Horizon experiment results saved to {horizon_results_file}")
    print("\nSummary of Horizon Experiment Results:")
    print(json.dumps(convert_np_types(horizon_results), indent=4))


def run_all_experiments():
    results = {}
    train_df, val_df, test_df = load_data()

    results["ARIMA"] = run_arima_experiment(train_df, val_df)
    results["Prophet"] = run_prophet_experiment(train_df, val_df)
    results["CustomTransformer"] = run_transformer_experiment(train_df, val_df)
    results["DartsTransformer"] = run_darts_transformer_experiment(train_df, val_df)

    results["global_system_metrics"] = log_system_metrics()
    results["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["hyperparameters"] = {
        "ARIMA": {"order": src.config.ARIMA_ORDER},
        "Prophet": {},
        "CustomTransformer": src.config.TRANSFORMER_CONFIG,
    }

    results_file = os.path.join(src.config.MODEL_SAVE_DIR, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(convert_np_types(results), f, indent=4)
    logging.info(f"Experiment results saved to {results_file}")

    run_horizon_experiments()


if __name__ == "__main__":
    run_all_experiments()
