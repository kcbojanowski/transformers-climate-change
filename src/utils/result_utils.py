import json
import numpy as np


def convert_np_types(data):
    if isinstance(data, dict):
        return {k: convert_np_types(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_np_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, np.generic):
        return data.item()
    else:
        return data


def load_results(results_path: str) -> dict:
    """Load experiment results from a JSON file and convert NumPy types to native Python types."""
    with open(results_path, "r") as f:
        results = json.load(f)
    print(results)
    return convert_np_types(results)


def unpack_results(results: dict) -> dict:
    unpacked = {}
    for model in ["ARIMA", "Prophet", "LSTM", "CustomTransformer"]:
        if model in results:
            unpacked[model] = results[model]
    unpacked["system_metrics"] = results.get("global_system_metrics", results.get("system_metrics"))
    unpacked["hyperparameters"] = results.get("hyperparameters", {})
    unpacked["timestamp"] = results.get("timestamp")
    return unpacked

def unpack_horizon_results(results: dict) -> dict:
    """
    Unpack the horizon experiment results from the provided dictionary.
    Returns a dictionary with keys for each model (ARIMA, Prophet, CustomTransformer)
    and additional keys for global system metrics, hyperparameters, and timestamp.
    """
    horizon_unpacked = {}
    for model in ["ARIMA", "Prophet", "CustomTransformer"]:
        if model in results:
            horizon_unpacked[model] = results[model]
    horizon_unpacked["system_metrics"] = results.get("global_system_metrics", results.get("system_metrics"))
    horizon_unpacked["hyperparameters"] = results.get("hyperparameters", {})
    horizon_unpacked["timestamp"] = results.get("timestamp")
    return horizon_unpacked

if __name__ == "__main__":
    results = load_results("../trained_models/experiment_results.json")
    unpacked = unpack_results(results)
    print("Dostępne klucze:", list(unpacked.keys()))
    print(unpacked)