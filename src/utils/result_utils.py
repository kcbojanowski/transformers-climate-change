import json
import numpy as np


def convert_np_types(obj):
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    else:
        return obj


def load_results(results_path: str) -> dict:
    """Load experiment results from a JSON file and convert NumPy types to native Python types."""
    with open(results_path, "r") as f:
        results = json.load(f)
    return convert_np_types(results)


def unpack_results(results: dict) -> dict:
    """
    Unpack the experiment results dictionary.

    Returns a dictionary with keys for each model (ARIMA, Prophet, LSTM, Transformer)
    and additional keys for system metrics, hyperparameters, and timestamp.
    """
    unpacked = {}
    for model in ["ARIMA", "Prophet", "CustomTransformer"]:
        if model in results:
            unpacked[model] = results[model]
    unpacked["system_metrics"] = results.get("global_system_metrics", results.get("system_metrics"))
    unpacked["hyperparameters"] = results.get("hyperparameters", {})
    unpacked["timestamp"] = results.get("timestamp")
    return unpacked