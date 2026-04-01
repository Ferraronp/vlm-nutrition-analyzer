import json
import logging
import requests
from groq import Groq

from config import GROQ_API_KEY, TEMPERATURE, MAX_COMPLETION_TOKENS, TOP_P
from prompts import LLM_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

AGENT_MODEL = "llama-3.3-70b-versatile"
MAX_AGENT_ITERATIONS = 10

# ---------------------------------------------------------------------------
# Tools: реализация
# ---------------------------------------------------------------------------

def search_food_db(query: str) -> dict:
    """Поиск продукта в Open Food Facts. Возвращает КБЖУ на 100г."""
    try:
        url = "https://world.openfoodfacts.org/cgi/search.pl"
        params = {
            "search_terms": query,
            "search_simple": 1,
            "action": "process",
            "json": 1,
            "page_size": 5,
            "fields": "product_name,nutriments"
        }
        resp = requests.get(url, params=params, timeout=8)
        resp.raise_for_status()
        data = resp.json()

        products = data.get("products", [])
        if not products:
            return {"found": False, "query": query}

        for product in products[:3]:
            n = product.get("nutriments", {})
            kcal = n.get("energy-kcal_100g") or n.get("energy-kcal")
            protein = n.get("proteins_100g")
            fat = n.get("fat_100g")
            carbs = n.get("carbohydrates_100g")

            if all(v is not None for v in [kcal, protein, fat, carbs]):
                name = (product.get("product_name") or query)[:40]
                return {"found": True, "name": name, "kcal": round(float(kcal), 1), "protein_g": round(float(protein), 1), "fat_g": round(float(fat), 1), "carbs_g": round(float(carbs), 1)}

        return {"found": False, "query": query, "reason": "No complete nutriment data"}

    except requests.RequestException as e:
        logger.warning(f"OFF API error for '{query}': {e}")
        return {"found": False, "query": query, "reason": str(e)}


def calculate_nutrition(kcal_per100g: float, protein_per100g: float,
                        fat_per100g: float, carbs_per100g: float,
                        amount_grams: float) -> dict:
    """Пересчитывает КБЖУ с 100г на указанное количество граммов."""
    factor = amount_grams / 100.0
    return {
        "amount_grams": amount_grams,
        "kcal": round(kcal_per100g * factor, 1),
        "protein_g": round(protein_per100g * factor, 1),
        "fat_g": round(fat_per100g * factor, 1),
        "carbs_g": round(carbs_per100g * factor, 1),
    }


# ---------------------------------------------------------------------------
# Tool schemas для Groq
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_food_db",
            "description": (
                "Search Open Food Facts database for a food item. "
                "Returns calories, protein, fat and carbohydrates per 100g. "
                "Use this for each ingredient mentioned in the VLM description."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Food item name in English, e.g. 'chicken breast', 'white rice cooked'"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_nutrition",
            "description": (
                "Calculate nutrition for a specific portion size given per-100g values. "
                "Use this after search_food_db to get actual values for the portion from the image."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "kcal_per100g":    {"type": "number", "description": "Calories per 100g"},
                    "protein_per100g": {"type": "number", "description": "Protein in grams per 100g"},
                    "fat_per100g":     {"type": "number", "description": "Fat in grams per 100g"},
                    "carbs_per100g":   {"type": "number", "description": "Carbohydrates in grams per 100g"},
                    "amount_grams":    {"type": "number", "description": "Actual portion size in grams"}
                },
                "required": ["kcal_per100g", "protein_per100g", "fat_per100g", "carbs_per100g", "amount_grams"]
            }
        }
    }
]

# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, args: dict) -> str:
    if name == "search_food_db":
        result = search_food_db(**args)
    elif name == "calculate_nutrition":
        result = calculate_nutrition(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, food_description: str) -> str:
        messages = [
            {"role": "system", "content": LLM_SYSTEM_PROMPT},
            {"role": "user",   "content": food_description}
        ]

        for iteration in range(MAX_AGENT_ITERATIONS):
            response = self.client.chat.completions.create(
                model=AGENT_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                temperature=TEMPERATURE,
                max_tokens=MAX_COMPLETION_TOKENS,
                top_p=TOP_P,
            )

            message = response.choices[0].message

            # Нет тул-коллов — агент дал финальный ответ
            if not message.tool_calls:
                return message.content or ""

            # Добавляем ответ ассистента с тул-коллами в историю
            messages.append({
                "role": "assistant",
                "content": message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in message.tool_calls
                ]
            })

            # Выполняем все тул-коллы и добавляем результаты
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                logger.info(f"[Agent] Tool call: {tc.function.name}({args})")
                result = dispatch_tool(tc.function.name, args)
                logger.info(f"[Agent] Tool result: {result}")

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result
                })

        logger.warning("Agent reached max iterations without final answer")
        return json.dumps({
            "clarification_required": "Не удалось завершить анализ за отведённое количество шагов."
        }, ensure_ascii=False)


# Global instance
llm_service = LLMService()