"""
Load Balancer for Food Nutrition Analysis API
Распределяет запросы между несколькими экземплярами API
"""

import asyncio
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import uuid
import random

from fastapi import FastAPI, HTTPException, Query, status, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import aiohttp
import async_timeout
from pydantic import BaseModel, HttpUrl, validator
from contextlib import asynccontextmanager

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Конфигурация
HEALTH_CHECK_INTERVAL = 30  # секунды
MAX_FAILED_ATTEMPTS = 3
REQUEST_TIMEOUT = 120  # секунды


class Instance(BaseModel):
    """Модель экземпляра API"""
    id: str
    url: HttpUrl
    cookies: Optional[Dict[str, str]] = None
    name: Optional[str] = None
    is_active: bool = True
    failed_attempts: int = 0
    last_check: Optional[datetime] = None
    created_at: datetime = datetime.now()

    class Config:
        schema_extra = {
            "example": {
                "url": "https://abc123.loca.lt",
                "cookies": {"colab-runtime-proxy-token": "token_value"},
                "name": "instance-1"
            }
        }


class InstanceCreate(BaseModel):
    """Модель для создания экземпляра"""
    url: HttpUrl
    cookies: Optional[Dict[str, str]] = None
    name: Optional[str] = None

    @validator('cookies')
    def validate_cookies(cls, v):
        if v and not isinstance(v, dict):
            raise ValueError('cookies must be a dictionary')
        return v


class FoodItem(BaseModel):
    name: str
    portion: str
    calories: float
    protein: float
    fat: float
    carbs: float

class Totals(BaseModel):
    calories: float
    protein: float
    fat: float
    carbs: float

class NutritionAnalysis(BaseModel):
    assumption: str
    items: list[FoodItem]
    totals: Totals
    disclaimer: str

class AnalysisData(BaseModel):
    analysis: NutritionAnalysis | dict  # допускаем как структурированный объект, так и уточняющий ответ (например, {"clarification_required": ...})
    vlm_output: str

class AnalysisResponse(BaseModel):
    status: str
    data: AnalysisData


class LoadBalancer:
    """Класс балансировщика нагрузки"""

    def __init__(self):
        self.instances: Dict[str, Instance] = {}
        self.current_index = 0
        self.session: Optional[aiohttp.ClientSession] = None

    async def start(self):
        """Запустить балансировщик и сессию aiohttp"""
        self.session = aiohttp.ClientSession()
        logger.info("Load balancer started")

    async def stop(self):
        """Остановить балансировщик и закрыть сессию"""
        if self.session:
            await self.session.close()
        logger.info("Load balancer stopped")

    def add_instance(self, instance_data: InstanceCreate) -> Instance:
        """Добавить новый экземпляр"""
        instance_id = str(uuid.uuid4())
        instance = Instance(
            id=instance_id,
            url=instance_data.url,
            cookies=instance_data.cookies,
            name=instance_data.name or f"instance-{len(self.instances) + 1}"
        )
        self.instances[instance_id] = instance
        logger.info(f"Added instance: {instance.name} ({instance.url})")
        return instance

    def remove_instance(self, instance_id: str) -> bool:
        """Удалить экземпляр"""
        if instance_id in self.instances:
            instance = self.instances.pop(instance_id)
            logger.info(f"Removed instance: {instance.name}")
            return True
        return False

    def get_active_instances(self) -> List[Instance]:
        """Получить список активных экземпляров"""
        return [inst for inst in self.instances.values() if inst.is_active]

    def get_next_instance(self) -> Optional[Instance]:
        """Получить следующий экземпляр по round-robin"""
        active_instances = self.get_active_instances()
        if not active_instances:
            return None

        # Round-robin выбор
        instance = active_instances[self.current_index % len(active_instances)]
        self.current_index += 1
        return instance

    async def check_instance_health(self, instance: Instance) -> bool:
        """Проверить состояние экземпляра"""
        if not self.session:
            return False

        try:
            health_url = f"{instance.url}health"
            async with async_timeout.timeout(REQUEST_TIMEOUT):
                async with self.session.get(health_url, cookies=instance.cookies) as response:
                    if response.status == 200:
                        instance.is_active = True
                        instance.failed_attempts = 0
                        instance.last_check = datetime.now()
                        logger.debug(f"Health check passed for {instance.name}")
                        return True
                    else:
                        raise Exception(f"Health check failed with status {response.status}")

        except Exception as e:
            instance.failed_attempts += 1
            instance.last_check = datetime.now()

            if instance.failed_attempts >= MAX_FAILED_ATTEMPTS:
                instance.is_active = False
                logger.warning(f"Instance {instance.name} marked as inactive after {MAX_FAILED_ATTEMPTS} failed attempts")
            else:
                logger.warning(f"Health check failed for {instance.name}: {str(e)}")

            return False

    async def health_check_loop(self):
        """Фоновая задача для проверки состояния экземпляров"""
        while True:
            try:
                tasks = []
                for instance in self.instances.values():
                    if instance.is_active:
                        tasks.append(self.check_instance_health(instance))

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")

            await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def forward_request(self, image_url: Optional[str] = None, image_file: Optional[UploadFile] = None) -> dict:
        """Переслать запрос на один из экземпляров"""
        instance = self.get_next_instance()
        if not instance:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No active instances available"
            )

        if not self.session:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Load balancer not initialized"
            )

        try:
            logger.info(f"Forwarding request to {instance.name}")

            # Если передан URL
            if image_url:
                target_url = f"{instance.url}/?image_url={image_url}"

                async with async_timeout.timeout(REQUEST_TIMEOUT):
                    async with self.session.get(target_url, cookies=instance.cookies) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"Request successful via {instance.name}")
                            return data
                        else:
                            error_text = await response.text()
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Instance returned error: {error_text}"
                            )

            # Если передан файл
            elif image_file:
                # Создаем form-data для пересылки файла
                form_data = aiohttp.FormData()

                # Читаем содержимое файла
                file_content = await image_file.read()

                # Добавляем файл в form-data
                form_data.add_field(
                    'file',
                    file_content,
                    filename=image_file.filename,
                    content_type=image_file.content_type or 'application/octet-stream'
                )

                target_url = f"{instance.url}upload"

                async with async_timeout.timeout(REQUEST_TIMEOUT):
                    async with self.session.post(target_url, cookies=instance.cookies, data=form_data) as response:
                        if response.status == 200:
                            data = await response.json()
                            logger.info(f"File upload request successful via {instance.name}")
                            return data
                        else:
                            error_text = await response.text()
                            raise HTTPException(
                                status_code=response.status,
                                detail=f"Instance returned error: {error_text}"
                            )
            else:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Either image_url or image_file must be provided"
                )

        except asyncio.TimeoutError:
            logger.error(f"Request timeout to {instance.name}")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="Request timeout"
            )
        except Exception as e:
            logger.error(f"Request failed to {instance.name}: {str(e)}")
            # Помечаем экземпляр как проблемный
            instance.failed_attempts += 1
            if instance.failed_attempts >= MAX_FAILED_ATTEMPTS:
                instance.is_active = False
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Failed to forward request: {str(e)}"
            )


# Глобальный экземпляр балансировщика
lb = LoadBalancer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await lb.start()
    # Запускаем фоновую задачу проверки состояния
    health_task = asyncio.create_task(lb.health_check_loop())
    logger.info("Load balancer API started")
    yield
    # Shutdown
    health_task.cancel()
    await lb.stop()
    logger.info("Load balancer API stopped")


app = FastAPI(
    title="Load Balancer for Food Nutrition Analysis API",
    description="Распределяет запросы между несколькими экземплярами API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/instances", response_model=Instance)
async def add_instance(instance_data: InstanceCreate):
    """Добавить новый экземпляр API"""
    try:
        instance = lb.add_instance(instance_data)
        return instance
    except Exception as e:
        logger.error(f"Failed to add instance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add instance: {str(e)}"
        )


@app.delete("/instances/{instance_id}")
async def remove_instance(instance_id: str):
    """Удалить экземпляр API"""
    if lb.remove_instance(instance_id):
        return {"message": "Instance removed successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found"
        )


@app.get("/instances")
async def list_instances():
    """Получить список всех экземпляров"""
    instances = list(lb.instances.values())
    return {
        "total": len(instances),
        "active": len(lb.get_active_instances()),
        "instances": instances
    }


@app.get("/instances/{instance_id}")
async def get_instance(instance_id: str):
    """Получить информацию о конкретном экземпляре"""
    instance = lb.instances.get(instance_id)
    if not instance:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Instance not found"
        )
    return instance


@app.get("/analyze", response_model=AnalysisResponse)
async def analyze_food_get(image_url: str = Query(..., description="URL of the food image")):
    """Анализировать изображение пищи через балансировщик (GET-запрос)"""
    try:
        result = await lb.forward_request(image_url=image_url)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze food: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze food: {str(e)}"
        )

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_food_post(
    image_url: str = Form(None, description="URL of the food image"),
    image_file: UploadFile = File(None, description="Food image file")
):
    """Анализировать изображение пищи через балансировщик (POST-запрос)"""
    if image_url == "string":
        image_url = None
    # Проверяем, что передан хотя бы один параметр
    if not image_url and not image_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either image_url or image_file must be provided"
        )
    
    # Проверяем, что передан только один параметр
    if image_url and image_file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only one of image_url or image_file should be provided"
        )
    
    try:
        result = await lb.forward_request(image_url=image_url, image_file=image_file)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze food: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze food: {str(e)}"
        )


@app.get("/health")
async def health_check():
    """Проверка работоспособности балансировщика"""
    active_count = len(lb.get_active_instances())
    total_count = len(lb.instances)
    
    return {
        "status": "healthy" if active_count > 0 else "degraded",
        "active_instances": active_count,
        "total_instances": total_count,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/stats")
async def get_stats():
    """Получить статистику балансировщика"""
    instances = list(lb.instances.values())
    
    stats = {
        "total_instances": len(instances),
        "active_instances": len([i for i in instances if i.is_active]),
        "inactive_instances": len([i for i in instances if not i.is_active]),
        "instances_with_cookies": len([i for i in instances if i.cookies]),
        "recent_failures": len([i for i in instances if i.failed_attempts > 0])
    }
    
    return stats


@app.get("/metrics/")
async def get_aggregated_metrics():
    """
    Собирает метрики со всех инстансов параллельно и агрегирует их.
    Использует куки каждого инстанса, так что работает с Colab-туннелями.
    Дашборд обращается только сюда — не знает про инстансы вообще.
    """
    instances = list(lb.instances.values())

    if not lb.session:
        raise HTTPException(status_code=503, detail="Load balancer session not started")

    # ── Параллельный опрос всех инстансов ────────────────────────────────────
    async def fetch_instance_metrics(inst: Instance) -> dict:
        try:
            url = f"{inst.url}metrics/"
            async with async_timeout.timeout(10):
                async with lb.session.get(url, cookies=inst.cookies) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return {"instance_id": inst.id, "name": inst.name, "ok": True, "data": data}
                    else:
                        return {"instance_id": inst.id, "name": inst.name, "ok": False, "error": f"HTTP {resp.status}"}
        except Exception as e:
            return {"instance_id": inst.id, "name": inst.name, "ok": False, "error": str(e)}

    results = await asyncio.gather(*[fetch_instance_metrics(i) for i in instances], return_exceptions=False)
    ok_results = [r for r in results if r["ok"]]
    failed_results = [r for r in results if not r["ok"]]

    # ── Агрегация ─────────────────────────────────────────────────────────────
    def safe_avg(values):
        vals = [v for v in values if v is not None]
        return sum(vals) / len(vals) if vals else None

    def safe_sum(values):
        vals = [v for v in values if v is not None]
        return sum(vals) if vals else None

    def safe_max(values):
        vals = [v for v in values if v is not None]
        return max(vals) if vals else None

    def safe_min(values):
        vals = [v for v in values if v is not None]
        return min(vals) if vals else None

    # Requests — суммируем по всем инстансам
    total_requests   = safe_sum([r["data"].get("requests", {}).get("total", 0)  for r in ok_results])
    total_errors     = safe_sum([r["data"].get("requests", {}).get("errors", 0) for r in ok_results])
    avg_latency_vals = [r["data"].get("requests", {}).get("avg_response_time_sec") for r in ok_results]
    min_latency_vals = [r["data"].get("requests", {}).get("min_response_time_sec") for r in ok_results]
    max_latency_vals = [r["data"].get("requests", {}).get("max_response_time_sec") for r in ok_results]
    p95_latency_vals = [r["data"].get("requests", {}).get("p95_response_time_sec") for r in ok_results]

    # System — усредняем CPU/RAM, суммируем диск
    cpu_vals          = [r["data"].get("system", {}).get("cpu_percent")                   for r in ok_results]
    ram_used_vals     = [r["data"].get("system", {}).get("memory", {}).get("used_percent") for r in ok_results]
    ram_total_vals    = [r["data"].get("system", {}).get("memory", {}).get("total_gb")     for r in ok_results]
    ram_avail_vals    = [r["data"].get("system", {}).get("memory", {}).get("available_gb") for r in ok_results]
    disk_total_vals   = [r["data"].get("system", {}).get("disk", {}).get("total_gb")       for r in ok_results]
    disk_free_vals    = [r["data"].get("system", {}).get("disk", {}).get("free_gb")        for r in ok_results]
    disk_used_p_vals  = [r["data"].get("system", {}).get("disk", {}).get("used_percent")   for r in ok_results]

    # GPU — собираем все GPU со всех инстансов (с пометкой источника)
    all_gpus = []
    for r in ok_results:
        gpus = r["data"].get("system", {}).get("gpu", [])
        if gpus:
            for g in gpus:
                all_gpus.append({**g, "_instance": r["name"]})

    aggregated = {
        "requests": {
            "total":                  total_requests,
            "errors":                 total_errors,
            "avg_response_time_sec":  safe_avg(avg_latency_vals),
            "min_response_time_sec":  safe_min(min_latency_vals),
            "max_response_time_sec":  safe_max(max_latency_vals),
            "p95_response_time_sec":  safe_avg(p95_latency_vals),  # avg of p95s
        },
        "system": {
            "cpu_percent": safe_avg(cpu_vals),
            "memory": {
                "used_percent":  safe_avg(ram_used_vals),
                "total_gb":      safe_sum(ram_total_vals),
                "available_gb":  safe_sum(ram_avail_vals),
            },
            "disk": {
                "total_gb":      safe_sum(disk_total_vals),
                "free_gb":       safe_sum(disk_free_vals),
                "used_percent":  safe_avg(disk_used_p_vals),
            },
            "gpu": all_gpus,
        },
        # Мета — состояние самого LB и отдельные данные каждого инстанса
        "lb": {
            "total_instances":    len(instances),
            "active_instances":   len([i for i in instances if i.is_active]),
            "instances_polled":   len(ok_results),
            "instances_failed":   len(failed_results),
            "failed_details":     [{"name": r["name"], "error": r["error"]} for r in failed_results],
        },
        "per_instance": [
            {
                "id":     r["instance_id"],
                "name":   r["name"],
                "ok":     r["ok"],
                "data":   r.get("data"),
                "error":  r.get("error"),
            }
            for r in results
        ],
    }

    return aggregated


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)