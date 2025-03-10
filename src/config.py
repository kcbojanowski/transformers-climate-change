from typing import Dict, Tuple, Union

DATA_PATH: str = "data/processed/climate_data.csv"
ARIMA_ORDER: Tuple[int, int, int] = (1, 1, 1)

LSTM_CONFIG: Dict[str, Union[int, float]] = {
    "input_dim": 1,
    "hidden_dim": 32,
    "num_layers": 2,
    "output_dim": 1,
    "dropout": 0.1
}

TRANSFORMER_CONFIG: Dict[str, Union[int, float]] = {
    "input_size": 1,
    "model_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.1
}

EXPERIMENT_CONFIG: Dict[str, int] = {
    "num_steps": 10,       # Number of prediction steps
    "log_interval": 1      # Interval of epochs to log metrics
}