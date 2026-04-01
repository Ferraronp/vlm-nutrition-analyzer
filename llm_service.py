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
    """Шаг 3: собрать финальный JSON. Для database-items считаем сами, LLM только для estimate."""

    # Считаем всё что нашли в базе — без участия LLM
    computed_items = []
    estimate_items = []

    for item in ingredients:
        eng_name = item.get("ingredient", "")
        grams = float(item.get("amount_grams", 100))
        db = db_results.get(eng_name, {})
        if db.get("found"):
            factor = grams / 100.0
            computed_items.append({
                "_eng": eng_name,
                "amount_grams": grams,
                "source": "database",
                "kcal":      round(db["kcal"] * factor, 1),
                "protein_g": round(db["protein_g"] * factor, 1),
                "fat_g":     round(db["fat_g"] * factor, 1),
                "carbs_g":   round(db["carbs_g"] * factor, 1),
            })
        else:
            estimate_items.append({
                "_eng": eng_name,
                "amount_grams": grams,
            })

    # Для estimate-items просим LLM дать КБЖУ — один вызов для всех сразу
    estimate_results = {}
    if estimate_items:
        est_prompt = (
            "For each food item below, estimate kcal, protein_g, fat_g, carbs_g for the given amount_grams.\n"
            "Return ONLY JSON: {\"items\": [{\"ingredient\": \"...\", \"kcal\": N, \"protein_g\": N, \"fat_g\": N, \"carbs_g\": N}]}\n\n"
            + json.dumps(estimate_items, ensure_ascii=False)
        )
        try:
            resp = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": est_prompt}],
                temperature=0,
                max_tokens=512,
                response_format={"type": "json_object"},
            )
            est_data = json.loads(resp.choices[0].message.content or "{}")
            for ei in est_data.get("items", []):
                estimate_results[ei.get("ingredient", "").lower()] = ei
        except Exception as e:
            logger.warning(f"Estimate LLM call failed: {e}")

    # Финальный вызов: переводим на русский и собираем JSON
    all_items_for_llm = []
    for ci in computed_items:
        all_items_for_llm.append({
            "ingredient": ci["_eng"],
            "amount_grams": ci["amount_grams"],
            "source": "database",
            "kcal": ci["kcal"], "protein_g": ci["protein_g"],
            "fat_g": ci["fat_g"], "carbs_g": ci["carbs_g"],
        })
    for ei in estimate_items:
        key = ei["_eng"].lower()
        nums = estimate_results.get(key, {})
        all_items_for_llm.append({
            "ingredient": ei["_eng"],
            "amount_grams": ei["amount_grams"],
            "source": "estimate",
            "kcal": nums.get("kcal", 0), "protein_g": nums.get("protein_g", 0),
            "fat_g": nums.get("fat_g", 0), "carbs_g": nums.get("carbs_g", 0),
        })

    user_msg = (
        "Translate ingredient names to Russian and assemble the final nutrition JSON.\n"
        "Use EXACTLY the kcal/protein_g/fat_g/carbs_g values provided — do NOT recalculate.\n\n"
        + json.dumps(all_items_for_llm, ensure_ascii=False)
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": LLM_SUMMARIZE_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
        max_tokens=1024,
        top_p=TOP_P,
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or "{}"

    # Проставляем source из наших данных — по порядку items
    source_order = [i["source"] for i in all_items_for_llm]
    try:
        data = json.loads(raw)
        items = data.get("items", [])
        for i, llm_item in enumerate(items):
            llm_item["source"] = source_order[i] if i < len(source_order) else "estimate"
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return raw


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
