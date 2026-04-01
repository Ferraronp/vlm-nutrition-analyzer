import json
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from groq import Groq

from config import GROQ_API_KEY, TEMPERATURE, TOP_P
from prompts import LLM_SYSTEM_PROMPT, LLM_SUMMARIZE_PROMPT

logger = logging.getLogger(__name__)

LLM_MODEL = "llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# OFF API
# ---------------------------------------------------------------------------

def search_food_db(query: str) -> dict:
    """Поиск продукта в Open Food Facts. Возвращает КБЖУ на 100г."""
    url = "https://world.openfoodfacts.org/cgi/search.pl"
    params = {
        "search_terms": query,
        "search_simple": 1,
        "action": "process",
        "json": 1,
        "page_size": 3,
        "fields": "product_name,nutriments"
    }
    headers = {"User-Agent": "VLM-Nutrition-Analyzer/1.0 (https://github.com/Ferraronp/vlm-nutrition-analyzer)"}

    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=4)
            if resp.status_code == 503:
                time.sleep(0.5 * (2 ** attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.RequestException:
            if attempt == 2:
                return {"found": False, "query": query, "reason": "request failed"}
            time.sleep(0.5 * (2 ** attempt))
    else:
        return {"found": False, "query": query, "reason": "Service unavailable after retries"}

    for product in (data.get("products") or [])[:3]:
        n = product.get("nutriments", {})
        kcal    = n.get("energy-kcal_100g") or n.get("energy-kcal")
        protein = n.get("proteins_100g")
        fat     = n.get("fat_100g")
        carbs   = n.get("carbohydrates_100g")
        if all(v is not None for v in [kcal, protein, fat, carbs]):
            name = (product.get("product_name") or query)[:40]
            return {
                "found": True,
                "name": name,
                "kcal":      round(float(kcal), 1),
                "protein_g": round(float(protein), 1),
                "fat_g":     round(float(fat), 1),
                "carbs_g":   round(float(carbs), 1),
            }

    return {"found": False, "query": query, "reason": "No complete nutriment data"}


def search_food_db_batch(queries: list) -> dict:
    """Параллельный поиск нескольких ингредиентов."""
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_query = {executor.submit(search_food_db, q): q for q in queries}
        for future in as_completed(future_to_query, timeout=15):
            q = future_to_query[future]
            try:
                results[q] = future.result()
            except Exception as e:
                results[q] = {"found": False, "query": q, "reason": str(e)}
    return results


# ---------------------------------------------------------------------------
# Step 1: Extract ingredients from VLM output via LLM
# ---------------------------------------------------------------------------

EXTRACT_PROMPT = """Extract all food ingredients from the text below.
Return ONLY a JSON array of objects, no markdown, no extra text.
Each object: {"ingredient": "name in English", "amount_grams": number}
If amount is unclear, estimate in grams (e.g. 1 piece rice cake ~ 25g, 1 sausage ~ 30g, 1 tbsp sauce ~ 15g).
Ignore non-quantifiable garnishes (sesame seeds sprinkled, possible broth, spices).

Text:
"""


def extract_ingredients(client: Groq, vlm_output: str) -> list:
    """Шаг 1: вытащить список ингредиентов с граммовкой через LLM."""
    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": EXTRACT_PROMPT + vlm_output}],
        temperature=0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    text = resp.choices[0].message.content or "[]"
    try:
        data = json.loads(text)
        # модель может вернуть {"ingredients": [...]} или просто [...]
        if isinstance(data, list):
            return data
        for v in data.values():
            if isinstance(v, list):
                return v
    except Exception:
        pass
    return []


# ---------------------------------------------------------------------------
# Step 2: Build final JSON from DB results
# ---------------------------------------------------------------------------

def build_nutrition_json(client: Groq, vlm_output: str, ingredients: list, db_results: dict) -> str:
    """Шаг 3: собрать финальный JSON через LLM с готовыми данными из базы."""
    db_summary = []
    for item in ingredients:
        name = item.get("ingredient", "")
        grams = item.get("amount_grams", 100)
        db = db_results.get(name, {})
        if db.get("found"):
            factor = grams / 100.0
            db_summary.append({
                "ingredient": name,
                "amount_grams": grams,
                "source": "database",
                "kcal":      round(db["kcal"] * factor, 1),
                "protein_g": round(db["protein_g"] * factor, 1),
                "fat_g":     round(db["fat_g"] * factor, 1),
                "carbs_g":   round(db["carbs_g"] * factor, 1),
            })
        else:
            db_summary.append({
                "ingredient": name,
                "amount_grams": grams,
                "source": "estimate",
            })

    user_msg = (
        "VLM description:\n" + vlm_output + "\n\n"
        "Nutrition data from database:\n" + json.dumps(db_summary, ensure_ascii=False) + "\n\n"
        "Return the final nutrition JSON."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_SUMMARIZE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=TEMPERATURE,
        max_tokens=1024,
        top_p=TOP_P,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content or "{}"


# ---------------------------------------------------------------------------
# LLM Service
# ---------------------------------------------------------------------------

class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, food_description: str) -> str:
        # Шаг 1: извлечь ингредиенты
        ingredients = extract_ingredients(self.client, food_description)
        logger.info(f"[Pipeline] Extracted {len(ingredients)} ingredients: {[i.get('ingredient') for i in ingredients]}")

        if not ingredients:
            return json.dumps({"clarification_required": "Не удалось определить ингредиенты из описания."}, ensure_ascii=False)

        # Шаг 2: параллельный поиск в OFF
        queries = [i.get("ingredient", "") for i in ingredients if i.get("ingredient")]
        db_results = search_food_db_batch(queries)
        found = sum(1 for v in db_results.values() if v.get("found"))
        logger.info(f"[Pipeline] OFF lookup: {found}/{len(queries)} found")

        # Шаг 3: собрать финальный JSON
        result = build_nutrition_json(self.client, food_description, ingredients, db_results)
        logger.info("[Pipeline] Final JSON built")
        return result


# Global instance
llm_service = LLMService()
