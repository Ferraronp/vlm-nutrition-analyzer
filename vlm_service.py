import asyncio
import logging
import base64
import requests
from io import BytesIO
from typing import List
from concurrent.futures import ThreadPoolExecutor
import time
import os

from model_loader import model_loader
from prompts import VLM_SYSTEM_PROMPT
from config import MAX_NEW_TOKENS, TEMPERATURE, TOP_P, ENABLE_BATCHING
from batch_manager import batch_manager

logger = logging.getLogger(__name__)


def image_url_to_base64(image_url: str) -> str:
    """Конвертирует изображение по URL (http/https или file) в base64."""
    if image_url.startswith("file://"):
        file_path = image_url[7:]  # удаляем "file://"
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Local file not found: {file_path}")
        with open(file_path, "rb") as f:
            image_data = f.read()
    elif image_url.startswith(("http://", "https://")):
        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_data = response.content
    else:
        raise ValueError(f"Unsupported URL scheme: {image_url}")

    return base64.b64encode(image_data).decode('utf-8')


class VLMService:
    def __init__(self):
        self.model = model_loader.load_model()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def generate_single(self, image_url: str) -> str:
        """Обработка одного изображения через llama.cpp."""
        start_time = time.time()
        try:
            image_base64 = image_url_to_base64(image_url)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": VLM_SYSTEM_PROMPT
                        }
                    ]
                }
            ]

            response = self.model.create_chat_completion(
                messages=messages,
                max_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                top_p=TOP_P,
            )

            output_text = response['choices'][0]['message']['content']
            print(time.time() - start_time)
            return output_text.strip()

        except Exception as e:
            logger.error(f"Error in generate_single: {e}")
            return f"Error: {str(e)}"

    def generate_batch(self, image_urls: List[str]) -> List[str]:
        """Последовательная обработка нескольких изображений."""
        start_time = time.time()
        results = []
        for url in image_urls:
            try:
                results.append(self.generate_single(url))
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append(f"Error: {str(e)}")
        logger.info(f"Processed batch of {len(image_urls)} images in {time.time() - start_time:.2f}s")
        return results

    async def generate(self, image_url: str) -> str:
        """Основной метод с поддержкой батчинга."""
        if ENABLE_BATCHING:
            try:
                return await batch_manager.add_request(image_url)
            except Exception as e:
                logger.error(f"Batch manager error: {e}")
                return await self._generate_direct(image_url)
        else:
            return await self._generate_direct(image_url)

    async def _generate_direct(self, image_url: str) -> str:
        """Асинхронная обёртка для прямого вызова."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_single, image_url)


# Глобальный экземпляр
vlm_service = VLMService()


async def process_batch_callback(batch_requests):
    """Callback для batch_manager."""
    try:
        image_urls = [req.image_url for req in batch_requests]
        results = vlm_service.generate_batch(image_urls)
        for req, res in zip(batch_requests, results):
            if not req.future.done():
                req.future.set_result(res)
    except Exception as e:
        logger.error(f"Batch callback error: {e}")
        for req in batch_requests:
            if not req.future.done():
                req.future.set_exception(e)


batch_manager.process_batch_callback = process_batch_callback