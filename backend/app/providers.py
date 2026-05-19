import os
from typing import Protocol

from .schemas import InferenceStrategy, ModelRoute, ProviderUsage


OPENAI_MODEL_NAMES = {
    ModelRoute.OPENAI_SMALL: "gpt-4.1-mini",
    ModelRoute.OPENAI_LARGE: "gpt-4.1",
}

GEMINI_MODEL_NAMES = {
    ModelRoute.GEMINI_SMALL: "gemini-2.5-flash-lite",
    ModelRoute.GEMINI_LARGE: "gemini-2.5-flash",
}


class Provider(Protocol):
    def generate(self, prompt: str, strategy: InferenceStrategy, max_completion_tokens: int) -> tuple[str, ProviderUsage, str]:
        ...


class MockProvider:
    def generate(self, prompt: str, strategy: InferenceStrategy, max_completion_tokens: int) -> tuple[str, ProviderUsage, str]:
        output = build_mock_output(prompt, strategy)
        prompt_tokens = max(1, int(len(prompt) / 4))
        completion_tokens = max(1, int(len(output) / 4))
        usage = ProviderUsage(
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
            estimatedCostUsd=0.0,
        )
        return output, usage, "mock"


class OpenAIProvider:
    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate(self, prompt: str, strategy: InferenceStrategy, max_completion_tokens: int) -> tuple[str, ProviderUsage, str]:
        model = OPENAI_MODEL_NAMES[strategy.modelRoute]
        response = self.client.responses.create(
            model=model,
            input=prompt,
            max_output_tokens=max_completion_tokens,
        )
        text = response.output_text
        usage_obj = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage_obj, "input_tokens", 0) or 0)
        completion_tokens = int(getattr(usage_obj, "output_tokens", 0) or 0)
        usage = ProviderUsage(
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
            estimatedCostUsd=0.0,
        )
        return text, usage, "openai"


class GeminiProvider:
    def __init__(self):
        from google import genai

        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, prompt: str, strategy: InferenceStrategy, max_completion_tokens: int) -> tuple[str, ProviderUsage, str]:
        model = gemini_model_name(strategy.modelRoute)
        response = self.client.models.generate_content(
            model=model,
            contents=prompt,
        )
        text = getattr(response, "text", "") or ""
        prompt_tokens = max(1, int(len(prompt) / 4))
        completion_tokens = max(1, int(len(text) / 4))
        usage = ProviderUsage(
            promptTokens=prompt_tokens,
            completionTokens=completion_tokens,
            totalTokens=prompt_tokens + completion_tokens,
            estimatedCostUsd=0.0,
        )
        return text, usage, "gemini"


def gemini_model_name(route: ModelRoute) -> str:
    if route == ModelRoute.GEMINI_SMALL:
        return os.getenv("GEMINI_SMALL_MODEL", GEMINI_MODEL_NAMES[route])
    if route == ModelRoute.GEMINI_LARGE:
        return os.getenv("GEMINI_LARGE_MODEL", GEMINI_MODEL_NAMES[route])
    return GEMINI_MODEL_NAMES[route]


def provider_for(strategy: InferenceStrategy, force_mock: bool = False) -> Provider:
    if force_mock or strategy.modelRoute == ModelRoute.LOCAL:
        return MockProvider()
    if strategy.modelRoute in {ModelRoute.OPENAI_SMALL, ModelRoute.OPENAI_LARGE} and os.getenv("OPENAI_API_KEY"):
        return OpenAIProvider()
    if strategy.modelRoute in {ModelRoute.GEMINI_SMALL, ModelRoute.GEMINI_LARGE} and os.getenv("GEMINI_API_KEY"):
        return GeminiProvider()
    return MockProvider()


def build_inference_prompt(user_prompt: str, strategy: InferenceStrategy) -> str:
    parts = [
        "You are an AI API cost optimization assistant.",
        f"Reasoning depth: {strategy.reasoningDepth.value}.",
        "Return a useful answer to the user's request.",
    ]
    if strategy.contextCompression:
        parts.append("Compress context mentally and focus on the essential facts before answering.")
    if strategy.verify:
        parts.append("Check the final answer for correctness before responding.")
    parts.append(f"User request:\n{user_prompt}")
    return "\n".join(parts)


def build_mock_output(prompt: str, strategy: InferenceStrategy) -> str:
    preview = prompt.strip().replace("\n", " ")[:180]
    return (
        f"[mock:{strategy.modelRoute.value}] Selected strategy: {strategy.reasoningDepth.value} reasoning, "
        f"verify={strategy.verify}, retry={strategy.retry.value}, compression={strategy.contextCompression}.\n"
        f"Request preview: {preview}"
    )
