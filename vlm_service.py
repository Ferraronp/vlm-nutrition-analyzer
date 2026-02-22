from qwen_vl_utils import process_vision_info
import torch
import asyncio
import logging
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor
import time

from model_loader import model_loader
from prompts import VLM_SYSTEM_PROMPT
from config import MAX_NEW_TOKENS, ENABLE_BATCHING
from batch_manager import batch_manager

logger = logging.getLogger(__name__)


class VLMService:
    def __init__(self):
        self.model, self.processor = model_loader.load_model()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    def generate_batch(self, image_urls: List[str]) -> List[str]:
        """Обработка нескольких изображений одновременно"""
        start_time = time.time()
        try:
            messages_batch = []
            
            # Подготовка сообщений для каждого изображения
            for image_url in image_urls:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image_url,
                            },
                            {"type": "text", "text": VLM_SYSTEM_PROMPT},
                        ],
                    }
                ]
                messages_batch.append(messages)
            
            # Подготовка inputs для батча
            texts = []
            images_inputs = []
            videos_inputs = []
            
            for messages in messages_batch:
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                image_inputs, video_inputs = process_vision_info(messages)
                texts.append(text)
                images_inputs.append(image_inputs)
                videos_inputs.append(video_inputs)

            flat_images = []
            for sublist in images_inputs:
                if sublist:
                    flat_images.extend(sublist)
            flat_videos = []
            for sublist in videos_inputs:
                if sublist:
                    flat_videos.extend(sublist)
            if not flat_videos:
                flat_videos = None
            if not flat_images:
                flat_images = None

            # Обработка батча
            inputs = self.processor(
                text=texts,
                images=flat_images,
                videos=flat_videos,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to("cuda")
            
            # Генерация для всего батча
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    use_cache=True,
                    do_sample=False
                )
            
            # Обработка результатов
            results = []
            for i, (in_ids, out_ids) in enumerate(zip(inputs.input_ids, generated_ids)):
                generated_ids_trimmed = out_ids[len(in_ids):]
                output_text = self.processor.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )
                results.append(output_text)
            
            logger.info(f"Processed batch of {len(image_urls)} images in {time.time() - start_time:.2f}s")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            # Fallback: обработка по одному
            return self._process_individually(image_urls)
    
    def _process_individually(self, image_urls: List[str]) -> List[str]:
        """Обработка изображений по одному (fallback)"""
        results = []
        for image_url in image_urls:
            try:
                result = self.generate_single(image_url)
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing single image {image_url}: {e}")
                results.append(f"Error: {str(e)}")
        return results
    
    def generate_single(self, image_url: str) -> str:
        """Обработка одного изображения (как в оригинале)"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": VLM_SYSTEM_PROMPT},
                ],
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            use_cache=True,
            do_sample=False
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return output_text[0]
    
    async def generate(self, image_url: str) -> str:
        """Основной метод генерации с поддержкой батчинга"""
        if ENABLE_BATCHING:
            try:
                # Используем batch_manager для сбора запросов
                result = await batch_manager.add_request(image_url)
                return result
            except Exception as e:
                logger.error(f"Error using batch manager: {e}")
                # Fallback на прямую обработку
                return await self._generate_direct(image_url)
        else:
            # Прямая обработка без батчинга
            return await self._generate_direct(image_url)
    
    async def _generate_direct(self, image_url: str) -> str:
        """Прямая обработка без батчинга (асинхронная обертка)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.generate_single, image_url)


# Global instance
vlm_service = VLMService()


async def process_batch_callback(batch_requests):
    """Callback для обработки батчей из batch_manager"""
    try:
        # Собираем URL изображений из запросов
        image_urls = [req.image_url for req in batch_requests]
        
        # Обрабатываем батч
        results = vlm_service.generate_batch(image_urls)
        
        # Распределяем результаты по запросам
        for request, result in zip(batch_requests, results):
            if not request.future.done():
                request.future.set_result(result)
                
    except Exception as e:
        logger.error(f"Error in batch callback: {e}")
        # Устанавливаем ошибку для всех запросов в батче
        for request in batch_requests:
            if not request.future.done():
                request.future.set_exception(e)


# Устанавливаем callback для batch_manager
batch_manager.process_batch_callback = process_batch_callback