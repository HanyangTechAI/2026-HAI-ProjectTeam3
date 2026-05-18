from abc import ABC, abstractmethod

from .pricing import estimate_completion_tokens, estimate_cost, reasoning_multiplier
from .schemas import (
    CostEstimate,
    InferenceStrategy,
    ModelRoute,
    PromptAnalysis,
    ReasoningDepth,
    RetryStrategy,
    RiskLevel,
    RouteCandidate,
    TaskType,
)


class Policy(ABC):
    @abstractmethod
    def choose(self, analysis: PromptAnalysis, max_completion_tokens: int, force_mock: bool = False) -> tuple[InferenceStrategy, list[RouteCandidate]]:
        raise NotImplementedError


class RuleBasedPolicy(Policy):
    def choose(self, analysis: PromptAnalysis, max_completion_tokens: int, force_mock: bool = False) -> tuple[InferenceStrategy, list[RouteCandidate]]:
        candidates = build_candidates(analysis, max_completion_tokens)

        if force_mock:
            route = ModelRoute.LOCAL
            depth = ReasoningDepth.NONE
            verify = False
            retry = RetryStrategy.NONE
            compress = analysis.promptTokensEstimate > 1200
            reason = "forceMock=true so the request is handled locally with no API cost."
        elif analysis.riskLevel == RiskLevel.HIGH:
            route = ModelRoute.OPENAI_LARGE
            depth = ReasoningDepth.LONG
            verify = True
            retry = RetryStrategy.ONCE
            compress = analysis.promptTokensEstimate > 1200
            reason = "High-risk request: quality and verification are prioritized over cost."
        elif analysis.taskType == TaskType.CODING and analysis.complexityLevel == "high":
            route = ModelRoute.OPENAI_LARGE
            depth = ReasoningDepth.LONG
            verify = True
            retry = RetryStrategy.ONCE
            compress = analysis.promptTokensEstimate > 1200
            reason = "Complex coding task: a stronger model and verification reduce failure risk despite higher cost."
        elif analysis.taskType == TaskType.MATH and analysis.complexityLevel in {"medium", "high"}:
            route = ModelRoute.GEMINI_LARGE
            depth = ReasoningDepth.LONG
            verify = True
            retry = RetryStrategy.ONCE
            compress = False
            reason = "Math with non-trivial reasoning: choose a large Gemini route for lower estimated cost than the large OpenAI route while keeping verification."
        elif analysis.taskType == TaskType.STOCK:
            route = ModelRoute.OPENAI_LARGE
            depth = ReasoningDepth.LONG
            verify = True
            retry = RetryStrategy.ONCE
            compress = analysis.promptTokensEstimate > 1200
            reason = "Stock or investing request: use a stronger verified route because financial guidance is sensitive and often needs careful caveats."
        elif analysis.promptTokensEstimate > 1800:
            route = ModelRoute.GEMINI_SMALL
            depth = ReasoningDepth.SHORT
            verify = analysis.riskLevel != RiskLevel.LOW
            retry = RetryStrategy.NONE
            compress = True
            reason = "Long low-to-medium risk prompt: use context compression and a low-cost Gemini small route to control input cost."
        elif analysis.taskType in {TaskType.SUMMARIZATION, TaskType.CLASSIFICATION} and analysis.riskLevel == RiskLevel.LOW:
            route = ModelRoute.GEMINI_SMALL
            depth = ReasoningDepth.NONE
            verify = False
            retry = RetryStrategy.NONE
            compress = analysis.promptTokensEstimate > 900
            reason = "Low-risk summarization/classification usually needs less reasoning, so the cheapest capable route is selected."
        elif analysis.taskType == TaskType.WRITING and analysis.complexityLevel != "high":
            route = ModelRoute.OPENAI_SMALL
            depth = ReasoningDepth.SHORT
            verify = False
            retry = RetryStrategy.NONE
            compress = False
            reason = "Writing quality benefits from OpenAI small while staying far cheaper than a large model."
        elif analysis.complexityLevel == "high":
            route = ModelRoute.GEMINI_LARGE
            depth = ReasoningDepth.LONG
            verify = True
            retry = RetryStrategy.ONCE
            compress = analysis.promptTokensEstimate > 1200
            reason = "High complexity but not high risk: choose Gemini large as a cost-conscious strong route with verification."
        else:
            route = ModelRoute.OPENAI_SMALL
            depth = analysis.reasoningNeed
            verify = False
            retry = RetryStrategy.NONE
            compress = False
            reason = "Default low/medium complexity route: OpenAI small balances response quality and predictable low cost."

        strategy = InferenceStrategy(
            reasoningDepth=depth,
            modelRoute=route,
            verify=verify,
            retry=retry,
            contextCompression=compress,
            decisionReason=reason,
        )
        return strategy, candidates


def build_candidates(analysis: PromptAnalysis, max_completion_tokens: int) -> list[RouteCandidate]:
    depth = analysis.reasoningNeed
    completion_tokens = estimate_completion_tokens(
        prompt_tokens=analysis.promptTokensEstimate,
        max_completion_tokens=max_completion_tokens,
        reasoning_multiplier=reasoning_multiplier(depth.value),
    )
    rows: list[RouteCandidate] = []
    for route in ModelRoute:
        if route == ModelRoute.LOCAL:
            quality = "demo-only"
            tradeoff = "No API cost, but response quality is not representative."
        elif route.value.endswith("large"):
            quality = "high"
            tradeoff = "Higher expected quality for complex or risky requests, with higher cost."
        else:
            quality = "standard"
            tradeoff = "Lower cost for routine requests, with lower margin on complex tasks."

        rows.append(
            RouteCandidate(
                modelRoute=route,
                estimatedCost=estimate_cost(route, analysis.promptTokensEstimate, completion_tokens),
                expectedQuality=quality,
                tradeoff=tradeoff,
            )
        )
    return sorted(rows, key=lambda row: row.estimatedCost.totalCostUsd)


def selected_cost(analysis: PromptAnalysis, strategy: InferenceStrategy, max_completion_tokens: int) -> CostEstimate:
    completion_tokens = estimate_completion_tokens(
        prompt_tokens=analysis.promptTokensEstimate,
        max_completion_tokens=max_completion_tokens,
        reasoning_multiplier=reasoning_multiplier(strategy.reasoningDepth.value),
    )
    return estimate_cost(strategy.modelRoute, analysis.promptTokensEstimate, completion_tokens)
