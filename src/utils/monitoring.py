# Listing 2: src/utils/monitoring.py

import time
import psutil
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

def log_system_metrics() -> Dict[str, Any]:
    """
    Collects and returns system metrics: CPU usage, memory usage, and GPU information.
    """
    metrics: Dict[str, Any] = {
        "cpu_usage": get_cpu_usage(),
        "memory_usage": get_memory_usage(),
        "gpu_usage": get_gpu_usage(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    return metrics