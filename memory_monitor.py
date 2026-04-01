import asyncio
import psutil
import torch
import time
from fastapi import APIRouter, HTTPException
from concurrent.futures import ThreadPoolExecutor

# pynvml — прямой биндинг к NVML, видит реальную VRAM включая GGUF/llama.cpp
# pip install nvidia-ml-py
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except Exception:
    NVML_AVAILABLE = False

# GPUtil как запасной вариант (только температура и load)
try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

router = APIRouter()

# Executor для блокирующих вызовов psutil/pynvml из async handlers
_monitor_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="monitor")

# Глобальные переменные из app.py
try:
    from app import request_times, request_count, error_count
except ImportError:
    request_times = []
    request_count = 0
    error_count = 0


def _get_nvml_gpu_info(device_index: int) -> dict:
    """
    Читает реальную VRAM через NVML — работает для любых процессов,
    включая llama.cpp/GGUF которые не используют PyTorch CUDA allocator.
    """
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem_info  = pynvml.nvmlDeviceGetMemoryInfo(handle)
        util_info = pynvml.nvmlDeviceGetUtilizationRates(handle)
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except Exception:
            temp = None
        try:
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
        except Exception:
            name = None

        total_gb = mem_info.total / (1024 ** 3)
        used_gb  = mem_info.used  / (1024 ** 3)
        free_gb  = mem_info.free  / (1024 ** 3)

        return {
            "device":                     device_index,
            "name":                       name,
            "memory_total_gb":            round(total_gb, 2),
            "memory_used_gb":             round(used_gb, 2),
            "memory_free_gb":             round(free_gb, 2),
            "memory_allocated_gb":        round(used_gb, 2),  # алиас для дашборда
            "memory_reserved_gb":         None,
            "memory_utilization_percent": round(used_gb / total_gb * 100, 1) if total_gb > 0 else 0,
            "gpu_utilization_percent":    util_info.gpu,
            "temperature_c":              temp,
            "source":                     "nvml",
        }
    except Exception as e:
        return {"device": device_index, "error": str(e), "source": "nvml"}


def _collect_system_metrics() -> dict:
    """Синхронная часть — запускается в executor, не блокирует event loop."""
    cpu_pct = psutil.cpu_percent(interval=0.5)

    mem  = psutil.virtual_memory()
    disk = psutil.disk_usage('/')

    result = {
        "cpu_percent": cpu_pct,
        "memory": {
            "total_gb":     round(mem.total     / (1024 ** 3), 2),
            "available_gb": round(mem.available  / (1024 ** 3), 2),
            "used_gb":      round(mem.used       / (1024 ** 3), 2),
            "used_percent": mem.percent,
        },
        "disk": {
            "total_gb":     round(disk.total / (1024 ** 3), 1),
            "free_gb":      round(disk.free  / (1024 ** 3), 1),
            "used_percent": disk.percent,
        },
    }

    # GPU: NVML первый (видит GGUF VRAM), потом torch+GPUtil
    gpu_metrics = []

    if NVML_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                gpu_metrics.append(_get_nvml_gpu_info(i))
        except Exception:
            pass

    if not gpu_metrics and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props     = torch.cuda.get_device_properties(i)
            allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)
            reserved  = torch.cuda.memory_reserved(i)  / (1024 ** 3)
            total     = props.total_memory / (1024 ** 3)
            info = {
                "device":                     i,
                "name":                       props.name,
                "memory_total_gb":            round(total, 2),
                "memory_used_gb":             round(allocated, 2),
                "memory_allocated_gb":        round(allocated, 2),
                "memory_reserved_gb":         round(reserved, 2),
                "memory_utilization_percent": round(allocated / total * 100, 1) if total > 0 else 0,
                "temperature_c":              None,
                "gpu_utilization_percent":    None,
                "source":                     "torch",
            }
            if GPUTIL_AVAILABLE:
                try:
                    gpus = GPUtil.getGPUs()
                    if i < len(gpus):
                        info["temperature_c"]           = gpus[i].temperature
                        info["gpu_utilization_percent"] = round(gpus[i].load * 100, 1)
                except Exception:
                    pass
            gpu_metrics.append(info)

    if gpu_metrics:
        result["gpu"] = gpu_metrics

    return result


@router.get("/", summary="Get system metrics")
async def get_metrics():
    """Return system resource usage and request statistics."""

    loop = asyncio.get_event_loop()
    system = await loop.run_in_executor(_monitor_executor, _collect_system_metrics)

    times_snapshot = list(request_times)

    req_stats: dict = {
        "total":                   request_count,
        "errors":                  error_count,
        "avg_response_time_sec":   None,
        "min_response_time_sec":   None,
        "max_response_time_sec":   None,
        "p95_response_time_sec":   None,
    }

    if times_snapshot:
        sorted_times = sorted(times_snapshot)
        req_stats["avg_response_time_sec"] = round(sum(sorted_times) / len(sorted_times), 3)
        req_stats["min_response_time_sec"] = round(sorted_times[0], 3)
        req_stats["max_response_time_sec"] = round(sorted_times[-1], 3)
        if len(sorted_times) > 20:
            req_stats["p95_response_time_sec"] = round(
                sorted_times[int(len(sorted_times) * 0.95)], 3
            )

    vlm_info: dict = {}
    try:
        from batch_manager import batch_manager
        vlm_info = {
            "queue_size":     len(batch_manager.request_queue),
            "max_batch_size": batch_manager.max_batch_size,
            "is_running":     batch_manager.is_running,
        }
    except Exception:
        pass

    return {
        "requests": req_stats,
        "system":   system,
        **({"vlm": vlm_info} if vlm_info else {}),
    }


@router.get("/gpu", summary="Detailed GPU info")
async def get_gpu_info():
    """Return detailed GPU information."""
    if not NVML_AVAILABLE and not torch.cuda.is_available():
        raise HTTPException(status_code=404, detail="No GPU available")

    loop = asyncio.get_event_loop()
    system = await loop.run_in_executor(_monitor_executor, _collect_system_metrics)
    gpus = system.get("gpu")
    if not gpus:
        raise HTTPException(status_code=404, detail="No GPU detected")
    return {"gpus": gpus}
