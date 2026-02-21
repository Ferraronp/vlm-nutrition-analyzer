from fastapi import FastAPI, Query, HTTPException, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi import status
import time
import tempfile
import os
import uuid
from pathlib import Path
from contextlib import asynccontextmanager

from config import HOST, PORT, LOG_LEVEL
from vlm_service import vlm_service
from llm_service import llm_service
from memory_monitor import router as metrics_router

# Для сбора метрик запросов
request_times = []
request_count = 0
error_count = 0

# Временная папка для сохранения загруженных изображений
TEMP_IMAGE_DIR = Path("/tmp/food_nutrition")
TEMP_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# Поддерживаемые форматы изображений
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif'}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting Food Nutrition Analysis API...")
    yield
    # Shutdown
    print("Shutting down...")


app = FastAPI(
    title="Food Nutrition Analysis API",
    description="API for analyzing food images and calculating nutrition",
    version="1.0.0",
    lifespan=lifespan
)

# Подключаем роутер с метриками
app.include_router(metrics_router, prefix="/metrics", tags=["monitoring"])


def save_upload_file(upload_file: UploadFile) -> str:
    """Сохраняет загруженный файл во временную папку и возвращает путь к файлу."""
    # Проверка формата файла
    file_extension = Path(upload_file.filename).suffix.lower() if upload_file.filename else ''
    if file_extension not in SUPPORTED_IMAGE_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image format. Supported formats: {', '.join(SUPPORTED_IMAGE_FORMATS)}"
        )
    
    # Генерация уникального имени файла
    file_id = str(uuid.uuid4())
    file_path = TEMP_IMAGE_DIR / f"{file_id}{file_extension}"
    
    # Сохранение файла
    try:
        contents = upload_file.file.read()
        with open(file_path, 'wb') as f:
            f.write(contents)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error saving uploaded file"
        )
    finally:
        upload_file.file.close()
    
    return str(file_path)


def cleanup_temp_file(file_path: str):
    """Удаляет временный файл."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        print(f"Warning: Could not delete temporary file {file_path}: {e}")


async def process_image(image_source: str, is_file: bool = False):
    """Обрабатывает изображение и возвращает результат анализа."""
    global request_count, error_count, request_times
    
    request_count += 1
    start_time = time.time()
    
    try:
        # Генерация описания через VLM
        vlm_output = vlm_service.generate(image_source)
        
        # Анализ через LLM
        result = llm_service.generate(vlm_output)
        
        # Сохраняем время ответа
        elapsed = time.time() - start_time
        request_times.append(elapsed)
        # Оставляем только последние 1000 запросов
        if len(request_times) > 1000:
            request_times.pop(0)
        
        return {
            "status": "success",
            "data": {
                "analysis": result,
                "vlm_output": vlm_output  # можно убрать, если не нужно
            }
        }
        
    except HTTPException:
        # Пробрасываем HTTP исключения дальше
        raise
    except Exception as e:
        error_count += 1
        # Логируем ошибку
        print(f"Error processing request: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error: {str(e)}"
        )


@app.post("/upload", response_model=dict)
async def upload_and_analyze_food(file: UploadFile = File(...)):
    """
    Загрузить изображение блюда напрямую и получить анализ пищевой ценности.
    - file: изображение блюда (поддерживаемые форматы: jpg, jpeg, png, webp, bmp, gif)
    """
    # Сохраняем загруженный файл
    temp_file_path = save_upload_file(file)
    
    try:
        # Создаем file:// URL для передачи в VLM сервис
        file_url = f"file://{temp_file_path}"
        
        # Обрабатываем изображение
        result = await process_image(file_url, is_file=True)
        
        return result
        
    finally:
        # Удаляем временный файл
        cleanup_temp_file(temp_file_path)


@app.get("/", response_model=dict)
async def analyze_food(
    image_url: str = Query(None, description="URL of the food image"),
    image_file: UploadFile = File(None, description="Food image file")
):
    """
    Анализировать изображение блюда и получить информацию о пищевой ценности.
    Можно передать либо URL изображения, либо загрузить файл напрямую.
    
    - image_url: прямая ссылка на изображение блюда (опционально)
    - image_file: файл изображения блюда (опционально)
    """
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
    
    # Если передан URL
    if image_url:
        # Валидация URL
        if not image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image URL. Must start with http:// or https://"
            )
        
        return await process_image(image_url)
    
    # Если передан файл
    if image_file:
        # Сохраняем загруженный файл
        temp_file_path = save_upload_file(image_file)
        
        try:
            # Создаем file:// URL для передачи в VLM сервис
            file_url = f"file://{temp_file_path}"
            
            # Обрабатываем изображение
            result = await process_image(file_url, is_file=True)
            
            return result
            
        finally:
            # Удаляем временный файл
            cleanup_temp_file(temp_file_path)


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": time.time()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL
    )