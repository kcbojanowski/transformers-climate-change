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


def get_gpu_usage() -> dict[str, Any] | None:
    """
    Returns a list of dictionaries containing GPU information.
    Requires the GPUtil library.
    """
    if GPUtil:
        gpus = GPUtil.getGPUs()
        main_gpu = gpus[0]
        return {
            "memory_used": main_gpu.memoryUsed,
            "memory_total": main_gpu.memoryTotal,
            "temperature": main_gpu.temperature
        }
    else:
        return None


def log_system_metrics():
    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory().percent
    gpu = get_gpu_usage()
    metrics = {"cpu": cpu, "gpu": gpu, "memory": mem}
    return metrics