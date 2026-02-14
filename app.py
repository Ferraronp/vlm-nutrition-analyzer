from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

from config import HOST, PORT, LOG_LEVEL
from vlm_service import vlm_service
from llm_service import llm_service

app = FastAPI(title="Food Nutrition Analysis API")


@app.get("/")
async def home(image_url: str = Query(None)):
    if not image_url:
        return {"status": 419, "text": "no url"}

    try:
        vlm_output = vlm_service.generate(image_url)
        result = llm_service.generate(vlm_output)
        return {"status": 200, "text": result}
    except Exception as e:
        return {"status": 500, "text": f"Error: {str(e)}"}


@app.get("/health")
async def health_check():
    return {"status": 200, "text": "alive"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level=LOG_LEVEL
    )