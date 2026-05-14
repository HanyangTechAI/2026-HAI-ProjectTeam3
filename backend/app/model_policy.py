import json
import math
import os
from dataclasses import dataclass
from typing import Any

from .policy import Policy, RuleBasedPolicy, build_candidates
from .schemas import (
    InferenceStrategy,
    ModelRoute,
    PromptAnalysis,
    ReasoningDepth,
    RetryStrategy,
    RouteCandidate,
)


ACTION_SPECS: list[dict[str, Any]] = [
    {"route": "local", "depth": "none", "verify": False, "retry": "none", "compress": False},
    {"route": "gemini-small", "depth": "none", "verify": False, "retry": "none", "compress": False},
    {"route": "gemini-small", "depth": "short", "verify": False, "retry": "none", "compress": True},
    {"route": "openai-small", "depth": "short", "verify": False, "retry": "none", "compress": False},
    {"route": "gemini-large", "depth": "long", "verify": True, "retry": "once", "compress": False},
    {"route": "gemini-large", "depth": "long", "verify": True, "retry": "once", "compress": True},
    {"route": "openai-large", "depth": "long", "verify": True, "retry": "once", "compress": False},
    {"route": "openai-large", "depth": "long", "verify": True, "retry": "once", "compress": True},
]


def action_key(strategy: InferenceStrategy) -> str:
    return "|".join(
        [
            strategy.modelRoute.value,
            strategy.reasoningDepth.value,
            str(strategy.verify).lower(),
            strategy.retry.value,
            str(strategy.contextCompression).lower(),
        ]
    )


def spec_key(spec: dict[str, Any]) -> str:
    return "|".join(
        [
            spec["route"],
            spec["depth"],
            str(spec["verify"]).lower(),
            spec["retry"],
            str(spec["compress"]).lower(),
        ]
    )


ACTION_KEYS = [spec_key(spec) for spec in ACTION_SPECS]


@dataclass
class LinearRoutingModel:
    feature_names: list[str]
    action_keys: list[str]
    weights: list[list[float]]
    bias: list[float]

    def scores(self, features: list[float]) -> list[float]:
        rows = []
        for action_idx, row in enumerate(self.weights):
            rows.append(sum(w * x for w, x in zip(row, features)) + self.bias[action_idx])
        return rows

    def predict_index(self, features: list[float], blocked_actions: set[int] | None = None) -> int:
        blocked_actions = blocked_actions or set()
        scores = self.scores(features)
        best_idx = 0
        best_score = -math.inf
        for idx, score in enumerate(scores):
            if idx in blocked_actions:
                continue
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

    def to_json(self) -> dict[str, Any]:
        return {
            "model_type": "linear_routing_policy",
            "feature_names": self.feature_names,
            "action_keys": self.action_keys,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "LinearRoutingModel":
        return cls(
            feature_names=list(payload["feature_names"]),
            action_keys=list(payload["action_keys"]),
            weights=[[float(v) for v in row] for row in payload["weights"]],
            bias=[float(v) for v in payload["bias"]],
        )


def strategy_from_action_key(key: str, reason: str) -> InferenceStrategy:
    route, depth, verify, retry, compress = key.split("|")
    return InferenceStrategy(
        reasoningDepth=ReasoningDepth(depth),
        modelRoute=ModelRoute(route),
        verify=verify == "true",
        retry=RetryStrategy(retry),
        contextCompression=compress == "true",
        decisionReason=reason,
    )


def encode_features(analysis: PromptAnalysis, max_completion_tokens: int) -> tuple[list[str], list[float]]:
    names: list[str] = []
    values: list[float] = []

    def add(name: str, value: float):
        names.append(name)
        values.append(float(value))

    add("bias_input", 1.0)
    add("prompt_tokens_log", math.log1p(analysis.promptTokensEstimate) / 8.0)
    add("max_completion_log", math.log1p(max_completion_tokens) / 9.0)
    add("char_length_log", math.log1p(analysis.charLength) / 9.0)
    add("word_count_log", math.log1p(analysis.wordCount) / 8.0)
    add("complexity_score", analysis.complexityScore)
    add("risk_score", analysis.riskScore)

    for task in ["math", "coding", "general", "summarization", "writing", "classification"]:
        add(f"task_{task}", 1.0 if analysis.taskType.value == task else 0.0)
    for domain in ["business", "software", "education", "legal", "medical", "finance", "general"]:
        add(f"domain_{domain}", 1.0 if analysis.domain.value == domain else 0.0)
    for level in ["low", "medium", "high"]:
        add(f"risk_{level}", 1.0 if analysis.riskLevel.value == level else 0.0)
    for depth in ["none", "short", "long"]:
        add(f"reasoning_need_{depth}", 1.0 if analysis.reasoningNeed.value == depth else 0.0)
    for level in ["low", "medium", "high"]:
        add(f"complexity_{level}", 1.0 if analysis.complexityLevel == level else 0.0)

    return names, values


class ModelPolicy(Policy):
    def __init__(self, model: LinearRoutingModel, fallback: Policy | None = None):
        self.model = model
        self.fallback = fallback or RuleBasedPolicy()

    def choose(
        self,
        analysis: PromptAnalysis,
        max_completion_tokens: int,
        force_mock: bool = False,
    ) -> tuple[InferenceStrategy, list[RouteCandidate]]:
        if force_mock:
            return self.fallback.choose(analysis, max_completion_tokens, force_mock=True)

        candidates = build_candidates(analysis, max_completion_tokens)
        _, features = encode_features(analysis, max_completion_tokens)
        if len(features) != len(self.model.feature_names):
            return self.fallback.choose(analysis, max_completion_tokens, force_mock=False)

        blocked = {idx for idx, key in enumerate(self.model.action_keys) if key.startswith("local|")}
        idx = self.model.predict_index(features, blocked_actions=blocked)
        strategy = strategy_from_action_key(
            self.model.action_keys[idx],
            reason="Learned linear routing model selected this strategy from prompt analysis features.",
        )
        return strategy, candidates


def load_model_policy(path: str) -> ModelPolicy:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return ModelPolicy(LinearRoutingModel.from_json(payload))


def build_policy_from_env(default_path: str = "outputs/routing_policy.json") -> Policy:
    path = os.getenv("ROUTING_POLICY_PATH", default_path)
    if path and os.path.exists(path):
        return load_model_policy(path)
    return RuleBasedPolicy()
