# 🍽️ VLM Nutrition Analyzer

API-сервис для автоматического анализа питания по фотографии блюда. Загружаешь фото — получаешь КБЖУ по каждому ингредиенту.

## Как это работает

Трёхэтапный пайплайн обработки:

```
Фото блюда
    │
    ▼
[1] VLM (Qwen2.5-VL-7B)
    └─ Описывает состав и порции на изображении
    │
    ▼
[2] LLM (Llama 3.3 70B / Groq) + Open Food Facts API
    ├─ Извлекает список ингредиентов с граммовкой
    └─ Параллельный поиск в базе продуктов (до 5 потоков)
         ├─ Найдено в базе → точные данные из OFF
         └─ Не найдено → LLM оценивает КБЖУ самостоятельно
    │
    ▼
[3] LLM (Llama 3.3 70B / Groq)
    └─ Собирает финальный JSON с переводом на русский
    │
    ▼
JSON: ингредиенты, КБЖУ, источник данных (database / estimate)
```

Каждый элемент в ответе помечен `source`: `database` (данные из Open Food Facts) или `estimate` (оценка LLM).

## Стек

- **FastAPI** + Uvicorn — асинхронный веб-сервер
- **Qwen2.5-VL-7B-Instruct** — vision-language модель для анализа изображений (запускается локально на GPU)
- **Llama 3.3 70B** через Groq API — обработка текста и сборка итогового JSON
- **Open Food Facts** — бесплатная база нутриентов (~3 млн продуктов)
- **Батч-менеджер** — группирует входящие запросы для эффективной загрузки VLM
- **Load balancer** — распределяет нагрузку при нескольких Colab-инстансах
- **Memory monitor** — мониторинг GPU/CPU памяти через `/metrics/system`

## Требования

- Python 3.8+
- GPU с CUDA (рекомендуется для VLM; минимум 8 ГБ VRAM, 16+ ГБ для стабильной работы)
- Groq API ключ (бесплатный тир доступен на [console.groq.com](https://console.groq.com))
- 8+ ГБ RAM

## Быстрый старт

```bash
git clone https://github.com/Ferraronp/vlm-nutrition-analyzer
cd vlm-nutrition-analyzer

pip install -r requirements.txt

# Настройка окружения
echo "GROQ_API_KEY=your_key_here" > .env

# Запуск сервера
python app.py
```

Сервер поднимается на `http://localhost:8000`. Swagger UI: `http://localhost:8000/docs`.

## Запуск в Google Colab

Удобный вариант если нет локального GPU — использовать бесплатные Colab GPU.

1. Открыть [colab.ipynb](colab.ipynb) в Google Colab (выбрать GPU-рантайм T4)
2. Запустить все ячейки — сервер поднимется и выдаст публичный URL вида `https://8000-gpu-t4-XXXXXXX.prod.colab.dev`
3. Для доступа к API нужен cookie `colab-runtime-proxy-token` из DevTools браузера (вкладка Network → любой запрос → заголовки)

При нескольких Colab-инстансах запустить `load_balancer.py` — он проксирует запросы между ними.

## API

| Метод | Путь | Описание |
|---|---|---|
| `GET` | `/` | Анализ по URL изображения (`?image_url=...`) |
| `POST` | `/` | Анализ загруженного файла (multipart/form-data, поле `image_file`) |
| `POST` | `/upload` | То же, поле `file` |
| `GET` | `/health` | Проверка состояния |
| `GET` | `/metrics` | Статистика GPU/CPU/RAM |

Поддерживаемые форматы: JPG, JPEG, PNG, WebP, BMP, GIF. Максимальный размер: 10 МБ.

### Пример запроса

```bash
# По URL
curl "http://localhost:8000/?image_url=https://example.com/food.jpg"

# Загрузка файла
curl -X POST http://localhost:8000/upload \
     -F "file=@photo.jpg"
```

### Пример ответа

```json
{
  "status": "success",
  "data": {
    "analysis": {
      "items": [
        {
          "ingredient": "Рис белый варёный",
          "amount_grams": 200,
          "source": "database",
          "kcal": 260,
          "protein_g": 4.8,
          "fat_g": 0.6,
          "carbs_g": 57.2
        },
        {
          "ingredient": "Куриная грудка",
          "amount_grams": 150,
          "source": "estimate",
          "kcal": 165,
          "protein_g": 31.0,
          "fat_g": 3.6,
          "carbs_g": 0.0
        }
      ],
      "totals": {
        "kcal": 425,
        "protein_g": 35.8,
        "fat_g": 4.2,
        "carbs_g": 57.2
      }
    }
  }
}
```

## Конфигурация

Основные параметры в `config.py`:

| Параметр | По умолчанию | Описание |
|---|---|---|
| `MODEL_NAME` | `Qwen/Qwen2.5-VL-7B-Instruct` | VLM модель |
| `HOST` / `PORT` | `0.0.0.0:8000` | Адрес сервера |
| `ENABLE_BATCHING` | `True` | Батч-обработка запросов |
| `BATCH_TIMEOUT` | `2.0 сек` | Время сборки батча |
| `MAX_BATCH_SIZE` | `8` | Макс. запросов в батче |
| `MAX_NEW_TOKENS` | — | Лимит токенов VLM |
