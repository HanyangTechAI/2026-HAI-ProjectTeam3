import os
import re
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAILLMClient:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model_name: str):
        response = self.client.responses.create(
            model=model_name,
            input=prompt,
        )

        text = response.output_text

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }


class MockLLMClient:
    """Deterministic local client for demos and tests.

    It is intentionally small: it solves the built-in demo questions and a few
    simple arithmetic patterns without making network calls.
    """

    KNOWN_ANSWERS = {
        "mia has 12 stickers. she buys 8 more and gives 5 to her friend. how many stickers does mia have now?": "15",
        "a box has 6 rows of pencils with 4 pencils in each row. how many pencils are in the box?": "24",
        "sam read 18 pages on monday and twice as many pages on tuesday. how many pages did he read in total?": "54",
        "a jacket costs $80 and is discounted by 25 percent. what is the sale price?": "60",
        "there are 45 students. one third of them join the math club. how many students join the math club?": "15",
    }

    def generate(self, prompt: str, model_name: str):
        question = self._extract_question(prompt)
        answer = self._solve(question, prompt=prompt, model_name=model_name)
        reasoning = self._reasoning_text(prompt, answer)
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(reasoning.split()))

        return {
            "text": reasoning,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _extract_question(self, prompt: str) -> str:
        marker = "Question:"
        if marker in prompt:
            return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    def _solve(self, question: str, prompt: str, model_name: str) -> str:
        normalized = re.sub(r"\s+", " ", question.lower().strip())
        lowered = question.lower()

        if self._is_low_budget_small_model(prompt, model_name):
            if any(marker in lowered for marker in ["twice", "one third", "gives"]):
                numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", question.replace(",", ""))]
                if numbers:
                    return self._format_number(numbers[-1])

        if normalized in self.KNOWN_ANSWERS:
            return self.KNOWN_ANSWERS[normalized]

        numbers = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", question.replace(",", ""))]

        if not numbers:
            return "0"

        if "twice" in lowered and len(numbers) >= 1:
            return self._format_number(numbers[0] + 2 * numbers[0])
        if "one third" in lowered and len(numbers) >= 1:
            return self._format_number(numbers[0] / 3)
        if "percent" in lowered and "discount" in lowered and len(numbers) >= 2:
            return self._format_number(numbers[0] * (1.0 - numbers[1] / 100.0))
        if "each" in lowered or "rows" in lowered:
            if len(numbers) >= 2:
                return self._format_number(numbers[0] * numbers[1])
        if "gives" in lowered and len(numbers) >= 3:
            return self._format_number(numbers[0] + numbers[1] - numbers[2])
        if "more" in lowered and len(numbers) >= 2:
            return self._format_number(sum(numbers))

        return self._format_number(numbers[-1])

    def _reasoning_text(self, prompt: str, answer: str) -> str:
        if "Do not include any explanation" in prompt:
            return f"FINAL: {answer}"
        return f"I will compute the answer internally.\nFINAL: {answer}"

    def _format_number(self, value: float) -> str:
        if abs(value - round(value)) < 1e-9:
            return str(int(round(value)))
        return f"{value:.4f}".rstrip("0").rstrip(".")

    def _is_low_budget_small_model(self, prompt: str, model_name: str) -> bool:
        return model_name == "gpt-4.1-mini" and "very carefully" not in prompt


def build_llm_client(api_mode: str):
    if api_mode == "openai":
        return OpenAILLMClient()
    if api_mode == "mock":
        return MockLLMClient()

    raise ValueError(f"Unsupported api_mode: {api_mode}")
