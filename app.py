from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi import status
import time
from contextlib import asynccontextmanager

from config import HOST, PORT, LOG_LEVEL
from vlm_service import vlm_service
from llm_service import llm_service
from memory_monitor import router as metrics_router

# Для сбора метрик запросов
request_times = []
request_count = 0
error_count = 0


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


@app.get("/", response_model=dict)
async def analyze_food(image_url: str = Query(..., description="URL of the food image")):
    """
    Analyze food image and return nutritional information.
    - image_url: direct URL to the food image (required)
    """
    global request_count, error_count, request_times

    request_count += 1
    start_time = time.time()

    try:
        # Валидация URL (можно добавить)
        if not image_url.startswith(('http://', 'https://')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image URL. Must start with http:// or https://"
            )

        # Генерация описания через VLM
        vlm_output = vlm_service.generate(image_url)

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