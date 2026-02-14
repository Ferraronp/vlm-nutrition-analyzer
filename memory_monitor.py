import psutil
import torch
import time
from fastapi import APIRouter, HTTPException
from typing import Optional

# Попытка импорта GPUtil (необязательно)
try:
    import GPUtil

    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

router = APIRouter()

# Глобальные переменные для сбора статистики из app (если нужно)
# Их можно передавать через зависимости, но для простоты используем глобальный импорт
try:
    from app import request_times, request_count, error_count
except ImportError:
    # Заглушки для случая, если импорт не работает
    request_times = []
    request_count = 0
    error_count = 0


@router.get("/", summary="Get system metrics")
async def get_metrics():
    """Return system resource usage and request statistics."""
    metrics = {
        "requests": {
            "total": request_count,
            "errors": error_count,
            "avg_response_time_sec": 0,
            "requests_in_last_minute": 0
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory": {}
        }
    }

    # CPU memory
    mem = psutil.virtual_memory()
    metrics["system"]["memory"] = {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_percent": mem.percent
    }

    # Disk usage (for logging/debug)
    disk = psutil.disk_usage('/')
    metrics["system"]["disk"] = {
        "total_gb": disk.total / (1024 ** 3),
        "free_gb": disk.free / (1024 ** 3),
        "used_percent": disk.percent
    }

    # GPU metrics if available
    if torch.cuda.is_available():
        gpu_metrics = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            memory_allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            memory_reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)
            memory_total = gpu_props.total_memory / (1024 ** 3)

            gpu_info = {
                "device": i,
                "name": gpu_props.name,
                "memory_total_gb": memory_total,
                "memory_allocated_gb": memory_allocated,
                "memory_reserved_gb": memory_reserved,
                "memory_utilization_percent": (memory_allocated / memory_total) * 100 if memory_total > 0 else 0,
                "temperature_c": None
            }

            # Try to get temperature via GPUtil
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        gpu_info["temperature_c"] = gpus[i].temperature
                        gpu_info["gpu_utilization_percent"] = gpus[i].load * 100
                except:
                    pass

            gpu_metrics.append(gpu_info)

        metrics["system"]["gpu"] = gpu_metrics

    # Request statistics
    if request_times:
        metrics["requests"]["avg_response_time_sec"] = sum(request_times) / len(request_times)
        metrics["requests"]["min_response_time_sec"] = min(request_times)
        metrics["requests"]["max_response_time_sec"] = max(request_times)
        metrics["requests"]["p95_response_time_sec"] = sorted(request_times)[int(len(request_times) * 0.95)] if len(
            request_times) > 20 else None

        # Count requests in last minute (if timestamps were stored, but we don't have them)
        # For simplicity, we skip

    # VLM queue size (if using async processor)
    try:
        from async_processor import processor  # предполагаем, что есть глобальный processor
        metrics["vlm"] = {
            "queue_size": len(processor.request_queue) if hasattr(processor, 'request_queue') else 0,
            "max_batch_size": processor.max_batch_size if hasattr(processor, 'max_batch_size') else None
        }
    except (ImportError, AttributeError):
        pass

    return metrics


@router.get("/gpu", summary="Detailed GPU info")
async def get_gpu_info():
    """Return detailed GPU information if available."""
    if not torch.cuda.is_available():
        raise HTTPException(status_code=404, detail="No GPU available")

    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": props.total_memory / (1024 ** 3),
            "compute_capability": f"{props.major}.{props.minor}",
            "multi_processor_count": props.multi_processor_count
        })

    return {"gpus": gpus}