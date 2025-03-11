
#####################
#      PATHS        #
#####################
DATA_PATH = "../processed/full_climate_data.csv"
PROCESSED_TRAIN_DATA_PATH = "../processed/train_climate_data.csv"
PROCESSED_VAL_DATA_PATH = "../processed_/val_climate_data.csv"
PROCESSED_TEST_DATA_PATH = "../processed/test_climate_data.csv"
SNIPPET_TRAIN_DATA_PATH = "../processed/snippet_train_climate_data.csv"
SNIPPET_VAL_DATA_PATH = "../processed/snippet_val_climate_data.csv"
SNIPPET_TEST_DATA_PATH = "../processed/snippet_test_climate_data.csv"

#####################
#      CONFIGS      #
#####################
ARIMA_ORDER = (1, 1, 1)

LSTM_CONFIG = {
    "input_dim": 1,
    "hidden_dim": 32,
    "num_layers": 2,
    "output_dim": 1,
    "dropout": 0.1
}

TRANSFORMER_CONFIG = {
    "input_size": 1,
    "model_dim": 32,
    "num_heads": 4,
    "num_layers": 2,
    "output_size": 1,
    "dropout": 0.1
}

EXPERIMENT_CONFIG = {
    "num_steps": 10,       # Number of prediction steps
    "log_interval": 1      # Interval of epochs to log metrics
}