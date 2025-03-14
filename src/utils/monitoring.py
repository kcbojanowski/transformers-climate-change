import time
import psutil
import logging
from typing import Any, Dict, Optional, List

try:
    import GPUtil
except ImportError:
    GPUtil = None  # Install it via: pip install gputil

def get_cpu_usage() -> float:
    """
    Returns the current CPU usage percentage.
    """
    return psutil.cpu_percent(interval=1)

def get_memory_usage() -> float:
    """
    Returns the current RAM usage percentage.
    """
    memory = psutil.virtual_memory()
    return memory.percent

def get_gpu_usage() -> Optional[List[Dict[str, Any]]]:
    """
    Returns a list of dictionaries containing GPU information.
    Requires the GPUtil library.
    """
    if GPUtil:
        gpus = GPUtil.getGPUs()
        return [{
            "id": gpu.id,
            "name": gpu.name,
            "load": gpu.load * 100,
            "memory_used": gpu.memoryUsed,
            "memory_total": gpu.memoryTotal,
            "temperature": gpu.temperature
        } for gpu in gpus]
    else:
        return None

def log_system_metrics():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    metrics = {"cpu": cpu, "memory": mem}
    logging.info(f"System Metrics: {metrics}")
    return metrics