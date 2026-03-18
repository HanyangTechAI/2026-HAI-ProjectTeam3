import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class BaseLLMClient:
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError


class MockLLMClient(BaseLLMClient):
    def generate(self, prompt: str) -> LLMResponse:
        """
        테스트용 mock.
        실제 성능 검증은 안 되지만 파이프라인 확인 가능.
        """
        text = "FINAL: 42"
        return LLMResponse(
            text=text,
            prompt_tokens=max(1, len(prompt) // 4),
            completion_tokens=max(1, len(text) // 4),
            total_tokens=max(2, len(prompt) // 4 + len(text) // 4),
        )


class OpenAIChatClient(BaseLLMClient):
    def __init__(self, model_name: str, temperature: float, max_new_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, prompt: str) -> LLMResponse:
        response = self.client.responses.create(
            model=self.model_name,
            input=prompt,
            temperature=self.temperature,
            max_output_tokens=self.max_new_tokens,
        )

        text = response.output_text

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else 0

        return LLMResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )


def build_llm_client(api_mode: str, model_name: str, temperature: float, max_new_tokens: int) -> BaseLLMClient:
    if api_mode == "mock":
        return MockLLMClient()
    return OpenAIChatClient(
        model_name=model_name,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )