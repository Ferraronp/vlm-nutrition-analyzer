from groq import Groq

from config import GROQ_API_KEY, TEMPERATURE, MAX_COMPLETION_TOKENS, TOP_P, REASONING_EFFORT
from prompts import LLM_SYSTEM_PROMPT


class LLMService:
    def __init__(self):
        self.client = Groq(api_key=GROQ_API_KEY)

    def generate(self, food_description: str) -> str:
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[
                {
                    "role": "system",
                    "content": LLM_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": food_description
                }
            ],
            temperature=TEMPERATURE,
            max_completion_tokens=MAX_COMPLETION_TOKENS,
            top_p=TOP_P,
            reasoning_effort=REASONING_EFFORT,
            stream=False,
            stop=None
        )

        return str(completion.choices[0].message.content)


# Global instance
llm_service = LLMService()