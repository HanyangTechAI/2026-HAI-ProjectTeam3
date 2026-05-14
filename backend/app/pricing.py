from dataclasses import dataclass

from .schemas import CostEstimate, ModelRoute


@dataclass(frozen=True)
class ModelPrice:
    input_per_million: float
    output_per_million: float


MODEL_PRICES: dict[ModelRoute, ModelPrice] = {
    ModelRoute.OPENAI_SMALL: ModelPrice(input_per_million=0.15, output_per_million=0.60),
    ModelRoute.OPENAI_LARGE: ModelPrice(input_per_million=2.50, output_per_million=10.00),
    ModelRoute.GEMINI_SMALL: ModelPrice(input_per_million=0.10, output_per_million=0.40),
    ModelRoute.GEMINI_LARGE: ModelPrice(input_per_million=1.25, output_per_million=5.00),
    ModelRoute.LOCAL: ModelPrice(input_per_million=0.00, output_per_million=0.00),
}


def estimate_completion_tokens(prompt_tokens: int, max_completion_tokens: int, reasoning_multiplier: float) -> int:
    baseline = max(64, int(prompt_tokens * 0.35 * reasoning_multiplier))
    return min(max_completion_tokens, baseline)


def estimate_cost(route: ModelRoute, prompt_tokens: int, completion_tokens: int) -> CostEstimate:
    price = MODEL_PRICES[route]
    input_cost = prompt_tokens / 1_000_000 * price.input_per_million
    output_cost = completion_tokens / 1_000_000 * price.output_per_million
    return CostEstimate(
        promptTokens=prompt_tokens,
        completionTokens=completion_tokens,
        inputCostUsd=round(input_cost, 8),
        outputCostUsd=round(output_cost, 8),
        totalCostUsd=round(input_cost + output_cost, 8),
    )


def reasoning_multiplier(reasoning_depth: str) -> float:
    if reasoning_depth == "long":
        return 2.0
    if reasoning_depth == "short":
        return 1.35
    return 1.0
