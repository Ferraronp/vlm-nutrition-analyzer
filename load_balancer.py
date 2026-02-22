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


@app.get("/analyze")
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


@app.post("/analyze")
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)