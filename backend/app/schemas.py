from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskType(str, Enum):
    MATH = "math"
    CODING = "coding"
    GENERAL = "general"
    SUMMARIZATION = "summarization"
    WRITING = "writing"
    CLASSIFICATION = "classification"


class Domain(str, Enum):
    BUSINESS = "business"
    SOFTWARE = "software"
    EDUCATION = "education"
    LEGAL = "legal"
    MEDICAL = "medical"
    FINANCE = "finance"
    GENERAL = "general"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ReasoningDepth(str, Enum):
    NONE = "none"
    SHORT = "short"
    LONG = "long"


class ModelRoute(str, Enum):
    OPENAI_SMALL = "openai-small"
    OPENAI_LARGE = "openai-large"
    GEMINI_SMALL = "gemini-small"
    GEMINI_LARGE = "gemini-large"
    LOCAL = "local"


class RetryStrategy(str, Enum):
    NONE = "none"
    ONCE = "once"


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    maxCompletionTokens: int = Field(default=512, ge=32, le=8192)
    forceMock: bool = False


class PromptAnalysis(BaseModel):
    charLength: int
    wordCount: int
    promptTokensEstimate: int
    complexityScore: float
    complexityLevel: str
    domain: Domain
    taskType: TaskType
    reasoningNeed: ReasoningDepth
    riskScore: float
    riskLevel: RiskLevel
    signals: list[str]


class InferenceStrategy(BaseModel):
    reasoningDepth: ReasoningDepth
    modelRoute: ModelRoute
    verify: bool
    retry: RetryStrategy
    contextCompression: bool
    decisionReason: str


class CostEstimate(BaseModel):
    promptTokens: int
    completionTokens: int
    inputCostUsd: float
    outputCostUsd: float
    totalCostUsd: float
    currency: str = "USD"


class ProviderUsage(BaseModel):
    promptTokens: int
    completionTokens: int
    totalTokens: int
    estimatedCostUsd: float


class RouteCandidate(BaseModel):
    modelRoute: ModelRoute
    estimatedCost: CostEstimate
    expectedQuality: str
    tradeoff: str


class OptimizeResponse(BaseModel):
    requestId: str
    analysis: PromptAnalysis
    strategy: InferenceStrategy
    estimatedCost: CostEstimate
    candidates: list[RouteCandidate]
    output: str
    providerMode: str
    usage: ProviderUsage


class StatsResponse(BaseModel):
    totalRequests: int
    totalTokens: int
    estimatedCostUsd: float
    routeCounts: dict[str, int]
    taskCounts: dict[str, int]
    recent: list[dict[str, Any]]
