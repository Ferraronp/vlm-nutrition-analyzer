import asyncio
import time
import threading
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass, field
import logging

from config import BATCH_TIMEOUT, MAX_BATCH_SIZE, ENABLE_BATCHING

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Один запрос в батче"""
    image_url: str
    future: asyncio.Future
    timestamp: float = field(default_factory=time.time)


class BatchManager:
    """Менеджер для сбора запросов в батчи и их обработки"""
    
    def __init__(self, batch_timeout: float = BATCH_TIMEOUT, max_batch_size: int = MAX_BATCH_SIZE):
        self.batch_timeout = batch_timeout
        self.max_batch_size = max_batch_size
        self.request_queue: List[BatchRequest] = []
        self.queue_lock = threading.Lock()
        self.processing_task: Optional[asyncio.Task] = None
        self.is_running = False
        self.process_batch_callback: Optional[Callable[[List[BatchRequest]], None]] = None
        
    def start(self):
        """Запустить менеджер батчей"""
        if not ENABLE_BATCHING:
            logger.info("Batching is disabled")
            return
            
        self.is_running = True
        logger.info(f"BatchManager started with timeout={self.batch_timeout}s, max_size={self.max_batch_size}")
        
    def stop(self):
        """Остановить менеджер батчей"""
        self.is_running = False
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            
    async def add_request(self, image_url: str) -> str:
        """Добавить запрос в очередь на батч-обработку"""
        if not ENABLE_BATCHING:
            raise RuntimeError("Batching is disabled")
            
        # Создаем future для ожидания результата
        future = asyncio.Future()
        request = BatchRequest(image_url=image_url, future=future)
        
        # Добавляем в очередь
        with self.queue_lock:
            self.request_queue.append(request)
            
        # Запускаем обработку, если еще не запущена
        if self.processing_task is None or self.processing_task.done():
            self.processing_task = asyncio.create_task(self._process_batches())
            
        # Ждем результата
        try:
            result = await future
            return result
        except Exception as e:
            logger.error(f"Error waiting for batch result: {e}")
            raise
            
    async def _process_batches(self):
        """Основной цикл обработки батчей"""
        while self.is_running and self.request_queue:
            try:
                # Ждем, пока наберется достаточно запросов или пройдет таймаут
                await self._wait_for_batch()
                
                # Формируем батч для обработки
                batch = self._form_batch()
                if batch:
                    logger.info(f"Processing batch of {len(batch)} requests")
                    # Вызываем callback для обработки батча
                    if self.process_batch_callback:
                        await self.process_batch_callback(batch)
                        
            except asyncio.CancelledError:
                logger.info("Batch processing cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch processing: {e}")
                await asyncio.sleep(1)
                
    async def _wait_for_batch(self):
        """Ждем, пока наберется достаточно запросов или пройдет таймаут"""
        start_time = time.time()
        
        while self.is_running:
            with self.queue_lock:
                queue_size = len(self.request_queue)
                
            # Проверяем условия для обработки батча
            if queue_size >= self.max_batch_size:
                logger.debug(f"Batch size reached: {queue_size}/{self.max_batch_size}")
                break
                
            if queue_size > 0 and (time.time() - start_time) >= self.batch_timeout:
                logger.debug(f"Batch timeout reached: {time.time() - start_time:.1f}s")
                break
                
            if queue_size == 0:
                # Нет запросов, ждем немного
                await asyncio.sleep(0.1)
            else:
                # Есть запросы, но еще не готовы к обработке
                await asyncio.sleep(0.1)
                
    def _form_batch(self) -> List[BatchRequest]:
        """Формируем батч из доступных запросов"""
        with self.queue_lock:
            # Берем минимум из доступных и максимального размера батча
            batch_size = min(len(self.request_queue), self.max_batch_size)
            batch = self.request_queue[:batch_size]
            self.request_queue = self.request_queue[batch_size:]
            
        logger.info(f"Formed batch with {len(batch)} requests")
        return batch
        
    def get_queue_size(self) -> int:
        """Получить текущий размер очереди"""
        with self.queue_lock:
            return len(self.request_queue)
            
    def get_pending_requests(self) -> List[Dict]:
        """Получить информацию о pending запросах"""
        with self.queue_lock:
            return [
                {
                    "image_url": req.image_url,
                    "wait_time": time.time() - req.timestamp
                }
                for req in self.request_queue
            ]


# Глобальный экземпляр BatchManager
batch_manager = BatchManager()


def initialize_batch_manager():
    """Инициализировать глобальный BatchManager"""
    batch_manager.start()


def shutdown_batch_manager():
    """Остановить глобальный BatchManager"""
    batch_manager.stop()