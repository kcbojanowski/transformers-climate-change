import os
import time
import json
import logging
import psutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

from ..config import (
    TRAIN_DATA_PATH,
    VAL_DATA_PATH,
    TEST_DATA_PATH,
    MODEL_SAVE_DIR,
    ARIMA_ORDER,
    LSTM_CONFIG,
    TRANSFORMER_CONFIG
)

# Import models
from src.models.arima_model import train_arima_model, forecast_arima
from src.models.prophet_model import train_prophet_model, forecast_prophet
from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

# Setup logging for detailed output
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def log_system_metrics():
    """Log basic system resource usage."""
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    logging.info(f"System Metrics -- CPU Usage: {cpu}%, Memory Usage: {mem}%")


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute evaluation metrics."""
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    mape = np.mean(np.abs((actual - predicted) / (actual + 1e-8))) * 100
    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def load_data():
    """Load train, validation, and test datasets from disk."""
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df = pd.read_csv(VAL_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    return train_df, val_df, test_df


def run_arima_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train and evaluate ARIMA model."""
    logging.info("Starting ARIMA experiment...")
    start = time.time()
    # For ARIMA, aggregate the target variable (t2m) by daily mean over all locations.
    train_series = train_df.groupby('time_idx')['t2m'].mean()
    model = train_arima_model(train_series, order=ARIMA_ORDER)
    forecast_steps = len(val_df['time_idx'].unique())
    forecast = forecast_arima(model, steps=forecast_steps)
    elapsed = time.time() - start
    results = {"training_time": elapsed, "forecast_steps": forecast_steps}

    # Evaluate on validation set (aggregate true t2m)
    val_series = val_df.groupby('time_idx')['t2m'].mean().reindex(forecast.index)
    metrics = evaluate_forecast(val_series.values, forecast.values)
    results.update(metrics)
    logging.info(f"ARIMA results: {results}")

    # Save ARIMA model summary
    with open(os.path.join(MODEL_SAVE_DIR, "arima_model_summary.txt"), "w") as f:
        f.write(str(model.summary()))
    return results


def run_prophet_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train and evaluate Prophet model."""
    logging.info("Starting Prophet experiment...")
    start = time.time()
    # Prophet requires a DataFrame with 'ds' (dates) and 'y' (target)
    prophet_train = train_df.groupby('time_idx').mean().reset_index()
    prophet_train.rename(columns={'time': 'ds', 't2m': 'y'}, inplace=True)
    model = train_prophet_model(prophet_train)
    forecast_df = forecast_prophet(model, periods=len(val_df['time_idx'].unique()))
    elapsed = time.time() - start
    results = {"training_time": elapsed}

    prophet_val = val_df.groupby('time_idx').mean().reset_index()
    prophet_val.rename(columns={'time': 'ds', 't2m': 'y'}, inplace=True)
    forecast_yhat = forecast_df['yhat'].tail(len(prophet_val)).values
    metrics = evaluate_forecast(prophet_val['y'].values, forecast_yhat)
    results.update(metrics)
    logging.info(f"Prophet results: {results}")

    # Save Prophet model
    model.save(os.path.join(MODEL_SAVE_DIR, "prophet_model.pkl"))
    return results


def run_lstm_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train and evaluate LSTM model with detailed per-epoch logging."""
    logging.info("Starting LSTM experiment...")
    start = time.time()
    # Create sequences from aggregated training series (using t2m daily mean)
    sequence_length = 30
    series = train_df.groupby('time_idx')['t2m'].mean().values
    X, y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i + sequence_length])
        y.append(series[i + sequence_length])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Create a simple DataLoader (for demonstration, using full batch)
    train_loader = [(X_tensor, y_tensor)]
    val_loader = [(X_tensor, y_tensor)]  # using training data as validation for illustration

    lstm_model = LSTMModel(**LSTM_CONFIG)
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=LSTM_CONFIG["lr"])
    criterion = nn.MSELoss()

    num_epochs = LSTM_CONFIG["num_epochs"]
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs), desc="LSTM Training"):
        lstm_model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = lstm_model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss)

        # Validation (simple evaluation on the same data)
        lstm_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                x_batch, y_batch = batch
                outputs = lstm_model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
            val_losses.append(val_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    elapsed = time.time() - start
    results = {"training_time": elapsed, "train_loss": train_losses, "val_loss": val_losses}

    # Forecast using the last sequence from training data
    lstm_model.eval()
    with torch.no_grad():
        lstm_forecast = lstm_model(X_tensor[-1:]).detach().numpy().squeeze()
    target = y_tensor[-1].detach().numpy().squeeze()
    metrics = evaluate_forecast(np.array([target]), np.array([lstm_forecast]))
    results.update(metrics)
    logging.info(f"LSTM results: {results}")

    # Save LSTM model
    torch.save(lstm_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "lstm_model.pth"))
    return results


def run_transformer_experiment(train_df: pd.DataFrame, val_df: pd.DataFrame) -> dict:
    """Train and evaluate Transformer model with detailed per-epoch logging."""
    logging.info("Starting Transformer experiment...")
    start = time.time()
    # Prepare sequences (using same approach as for LSTM)
    sequence_length = 30
    series = train_df.groupby('time_idx')['t2m'].mean().values
    X, y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i + sequence_length])
        y.append(series[i + sequence_length])
    X = np.array(X).reshape(-1, sequence_length, 1)
    y = np.array(y).reshape(-1, 1)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    train_loader = [(X_tensor, y_tensor)]
    val_loader = [(X_tensor, y_tensor)]

    transformer_model = TransformerModel(**TRANSFORMER_CONFIG)
    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=TRANSFORMER_CONFIG["lr"])
    criterion = nn.MSELoss()

    num_epochs = TRANSFORMER_CONFIG["num_epochs"]
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs), desc="Transformer Training"):
        transformer_model.train()
        epoch_train_loss = 0.0
        for batch in train_loader:
            x_batch, y_batch = batch
            optimizer.zero_grad()
            outputs = transformer_model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        train_losses.append(epoch_train_loss)

        transformer_model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:
                x_batch, y_batch = batch
                outputs = transformer_model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
            val_losses.append(val_loss)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}: Train Loss: {epoch_train_loss:.4f}, Val Loss: {val_loss:.4f}")

    elapsed = time.time() - start
    results = {"training_time": elapsed, "train_loss": train_losses, "val_loss": val_losses}

    transformer_model.eval()
    with torch.no_grad():
        transformer_forecast = transformer_model(X_tensor[-1:]).detach().numpy().squeeze()
    target = y_tensor[-1].detach().numpy().squeeze()
    metrics = evaluate_forecast(np.array([target]), np.array([transformer_forecast]))
    results.update(metrics)
    logging.info(f"Transformer results: {results}")

    torch.save(transformer_model.state_dict(), os.path.join(MODEL_SAVE_DIR, "transformer_model.pth"))
    return results


def run_all_experiments():
    train_df, val_df, test_df = load_data()
    results = {}

    # Run ARIMA, Prophet, LSTM, and Transformer experiments
    results['ARIMA'] = run_arima_experiment(train_df, val_df)
    results['Prophet'] = run_prophet_experiment(train_df, val_df)
    results['LSTM'] = run_lstm_experiment(train_df, val_df)
    results['Transformer'] = run_transformer_experiment(train_df, val_df)

    log_system_metrics()

    results_file = os.path.join(MODEL_SAVE_DIR, "experiment_results.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=4)
    logging.info(f"Experiment results saved to {results_file}")

    print("\nSummary of Experiment Results:")
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    run_all_experiments()