import os
import re

from openai import OpenAI


class LLMResponse(dict):
    def __init__(self, text: str, prompt_tokens: int, completion_tokens: int, total_tokens: int):
        super().__init__(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        self.text = text
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class OpenAILLMClient:
    def __init__(
        self,
        default_model_name: str = "gpt-4.1-mini",
        temperature: float | None = None,
        max_new_tokens: int | None = None,
    ):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        self.client = OpenAI(api_key=api_key)
        self.default_model_name = default_model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str, model_name: str | None = None):
        kwargs = {
            "model": model_name or self.default_model_name,
            "input": prompt,
        }
        if self.temperature is not None:
            kwargs["temperature"] = self.temperature
        if self.max_new_tokens is not None:
            kwargs["max_output_tokens"] = self.max_new_tokens

        response = self.client.responses.create(**kwargs)

        text = response.output_text

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = prompt_tokens + completion_tokens

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


class MockLLMClient:
    """Deterministic local client for demos and tests.

    It is intentionally small: it solves the built-in math demo questions and
    returns plausible canned responses for general service demos.
    """

    KNOWN_ANSWERS = {
        "mia has 12 stickers. she buys 8 more and gives 5 to her friend. how many stickers does mia have now?": "15",
        "a box has 6 rows of pencils with 4 pencils in each row. how many pencils are in the box?": "24",
        "sam read 18 pages on monday and twice as many pages on tuesday. how many pages did he read in total?": "54",
        "a jacket costs $80 and is discounted by 25 percent. what is the sale price?": "60",
        "there are 45 students. one third of them join the math club. how many students join the math club?": "15",
    }

    def __init__(self, default_model_name: str = "gpt-4.1-mini"):
        self.default_model_name = default_model_name

    def generate(self, prompt: str, model_name: str | None = None):
        model_name = model_name or self.default_model_name
        question = self._extract_question(prompt)
        if not self._is_math_prompt(prompt):
            text = self._general_response(question, prompt)
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = max(1, len(text.split()))
            return LLMResponse(
                text=text,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            )

        answer = self._solve(question, prompt=prompt, model_name=model_name)
        reasoning = self._reasoning_text(prompt, answer)
        prompt_tokens = max(1, len(prompt.split()))
        completion_tokens = max(1, len(reasoning.split()))

        return LLMResponse(
            text=reasoning,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )

    def _extract_question(self, prompt: str) -> str:
        for marker in ["Request:", "Question:"]:
            if marker in prompt:
                return prompt.split(marker, 1)[1].strip()
        return prompt.strip()

    def _is_math_prompt(self, prompt: str) -> bool:
        return "FINAL: <number>" in prompt or "math word problem" in prompt

    def _general_response(self, request: str, prompt: str) -> str:
        lowered = request.lower()
        if "summar" in prompt.lower() or "요약" in lowered:
            return "Summary: The request asks for the key points to be condensed into a shorter, faithful version."
        if "classif" in prompt.lower() or "분류" in lowered:
            return "Label: general_request\nRationale: The input is best handled as a broad language task."
        if "rewrite" in prompt.lower() or any(word in lowered for word in ["email", "draft", "write", "rewrite", "작성", "고쳐"]):
            return "안녕하세요. 요청하신 내용을 바탕으로 명확하고 정중한 문장으로 정리했습니다."
        return "요청을 확인했습니다. 핵심 내용을 기준으로 간결하고 실용적인 답변을 제공합니다."

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


def build_llm_client(
    api_mode: str,
    model_name: str = "gpt-4.1-mini",
    temperature: float | None = None,
    max_new_tokens: int | None = None,
):
    if api_mode == "openai":
        return OpenAILLMClient(
            default_model_name=model_name,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
    if api_mode == "mock":
        return MockLLMClient(default_model_name=model_name)

    raise ValueError(f"Unsupported api_mode: {api_mode}")
