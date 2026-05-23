import argparse
import json
import math
import os
from typing import Any

from backend.app.model_policy import ACTION_KEYS, LinearRoutingModel, action_key, encode_features
from backend.app.policy import selected_cost
from backend.app.schemas import (
    InferenceStrategy,
    ModelRoute,
    PromptAnalysis,
    ReasoningDepth,
    RetryStrategy,
    RiskLevel,
    TaskType,
)
from backend.app.store import UsageStore


def softmax(scores: list[float], temperature: float) -> list[float]:
    scaled = [score / max(temperature, 1e-6) for score in scores]
    peak = max(scaled)
    values = [math.exp(score - peak) for score in scaled]
    total = sum(values)
    return [value / total for value in values]


def dot(row: list[float], features: list[float], bias: float) -> float:
    return sum(w * x for w, x in zip(row, features)) + bias


def normalize_prompt_text(prompt: str) -> str:
    return " ".join(str(prompt).strip().lower().split())


def load_source_task_types(paths: list[str]) -> dict[str, str]:
    task_types: dict[str, str] = {}
    for path in paths:
        if not path or not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            continue
        for item in payload:
            if not isinstance(item, dict):
                continue
            prompt = item.get("request") or item.get("question") or item.get("prompt")
            task_type = item.get("task_type") or item.get("category") or ""
            if prompt and task_type:
                task_types[normalize_prompt_text(str(prompt))] = str(task_type)
    return task_types


def source_task_type_for_payload(payload: dict[str, Any], source_task_types: dict[str, str]) -> str:
    training_context = payload.get("trainingContext", {})
    prompt = (
        training_context.get("prompt")
        or payload.get("prompt")
        or payload.get("request")
        or payload.get("question")
        or ""
    )
    if not prompt:
        return ""
    return source_task_types.get(normalize_prompt_text(str(prompt)), "")


def unsafe_route_penalty(analysis: PromptAnalysis, strategy: InferenceStrategy) -> tuple[float, list[str]]:
    penalty = 0.0
    reasons: list[str] = []
    is_large_route = strategy.modelRoute in {ModelRoute.OPENAI_LARGE, ModelRoute.GEMINI_LARGE}

    if analysis.riskLevel == RiskLevel.HIGH:
        if not is_large_route:
            penalty += 0.25
            reasons.append("high risk routed to a small model")
        if not strategy.verify:
            penalty += 0.20
            reasons.append("high risk without verification")
        if strategy.retry.value == "none":
            penalty += 0.10
            reasons.append("high risk without retry")

    if analysis.taskType == TaskType.STOCK:
        if not is_large_route:
            penalty += 0.25
            reasons.append("stock task routed to a small model")
        if not strategy.verify:
            penalty += 0.15
            reasons.append("stock task without verification")
        if strategy.retry.value == "none":
            penalty += 0.05
            reasons.append("stock task without retry")

    if analysis.taskType == TaskType.MATH and analysis.complexityLevel in {"medium", "high"}:
        if not is_large_route:
            penalty += 0.20
            reasons.append("medium/high math routed to a small model")
        if not strategy.verify:
            penalty += 0.14
            reasons.append("medium/high math without verification")
        if strategy.retry.value == "none":
            penalty += 0.06
            reasons.append("medium/high math without retry")

    if analysis.taskType == TaskType.CODING and analysis.complexityLevel == "high":
        if not is_large_route:
            penalty += 0.20
            reasons.append("high-complexity coding routed to a small model")
        if not strategy.verify:
            penalty += 0.14
            reasons.append("high-complexity coding without verification")
        if strategy.retry.value == "none":
            penalty += 0.06
            reasons.append("high-complexity coding without retry")

    return min(0.65, penalty), reasons


UNSAFE_SMALL_ACTION_KEYS = [
    "gemini-small|none|false|none|false",
    "gemini-small|short|false|none|true",
    "openai-small|short|false|none|false",
]


EASY_TASK_LARGE_ACTION_KEYS = [
    "gemini-large|long|true|once|false",
    "gemini-large|long|true|once|true",
    "openai-large|long|true|once|false",
    "openai-large|long|true|once|true",
]


def clamp_reward(value: float) -> float:
    return max(0.0, min(1.0, value))


def strategy_from_action_key(key: str) -> InferenceStrategy:
    route, reasoning, verify, retry, compression = key.split("|")
    return InferenceStrategy(
        modelRoute=ModelRoute(route),
        reasoningDepth=ReasoningDepth(reasoning),
        verify=verify == "true",
        retry=RetryStrategy(retry),
        contextCompression=compression == "true",
        decisionReason="Training action reconstructed from action key.",
    )


def cost_penalty(
    cost_usd: float,
    cost_penalty_weight: float,
    cost_reference_usd: float,
) -> float:
    if cost_penalty_weight <= 0.0:
        return 0.0
    reference = max(cost_reference_usd, 1e-9)
    return min(cost_penalty_weight, cost_penalty_weight * max(0.0, cost_usd) / reference)


def action_cost_usd(ex: dict[str, Any], key: str) -> float:
    analysis = PromptAnalysis.model_validate(ex["analysis"])
    strategy = strategy_from_action_key(key)
    return selected_cost(analysis, strategy, int(ex.get("max_completion_tokens", 512))).totalCostUsd


def safety_fallback_action_key(
    analysis_task_type: str,
    complexity_level: str,
    risk_level: str,
    source_task_type: str = "",
) -> str | None:
    if risk_level == RiskLevel.HIGH.value:
        return "openai-large|long|true|once|false"
    if source_task_type in {"easy_arithmetic", "ambiguous_tricky", "multi_step"}:
        return None
    if source_task_type == "ratio_percent_discount":
        return "gemini-large|long|true|once|false"
    if source_task_type == TaskType.STOCK.value:
        return "openai-large|long|true|once|false"
    if analysis_task_type == TaskType.STOCK.value:
        return "openai-large|long|true|once|false"
    if analysis_task_type == TaskType.MATH.value and complexity_level in {"medium", "high"}:
        return "gemini-large|long|true|once|false"
    if analysis_task_type == TaskType.CODING.value and complexity_level == "high":
        return "openai-large|long|true|once|false"
    return None


def easy_task_preferred_action_key(
    analysis_task_type: str,
    complexity_level: str,
    risk_level: str,
    source_task_type: str = "",
) -> str | None:
    if source_task_type in {"easy_arithmetic", "ambiguous_tricky", "multi_step"}:
        return "openai-small|short|false|none|false"
    if source_task_type in {TaskType.CLASSIFICATION.value, TaskType.SUMMARIZATION.value} and risk_level == RiskLevel.LOW.value:
        return "gemini-small|none|false|none|false"
    if source_task_type in {TaskType.GENERAL.value, TaskType.WRITING.value, TaskType.CODING.value} and risk_level == RiskLevel.LOW.value:
        return "openai-small|short|false|none|false"
    if risk_level != RiskLevel.LOW.value:
        return None
    if complexity_level not in {"low", "medium"}:
        return None
    if analysis_task_type in {TaskType.GENERAL.value, TaskType.WRITING.value, TaskType.CODING.value, TaskType.MATH.value}:
        return "openai-small|short|false|none|false"
    if analysis_task_type in {TaskType.CLASSIFICATION.value, TaskType.SUMMARIZATION.value}:
        return "gemini-small|none|false|none|false"
    return None


def augment_with_synthetic_routing_examples(
    examples: list[dict[str, Any]],
    negative_reward: float,
    positive_reward: float,
    easy_positive_reward: float,
    easy_large_negative_reward: float,
    cost_penalty_weight: float,
    cost_reference_usd: float,
) -> list[dict[str, Any]]:
    augmented = list(examples)
    negative_reward = max(0.0, min(1.0, negative_reward))
    positive_reward = max(0.0, min(1.0, positive_reward))
    easy_positive_reward = max(0.0, min(1.0, easy_positive_reward))
    easy_large_negative_reward = max(0.0, min(1.0, easy_large_negative_reward))

    for ex in examples:
        safe_key = safety_fallback_action_key(
            analysis_task_type=str(ex["analysis_task_type"]),
            complexity_level=str(ex["complexity_level"]),
            risk_level=str(ex["risk_level"]),
            source_task_type=str(ex.get("source_task_type", "")),
        )
        if safe_key is not None:
            for unsafe_key in UNSAFE_SMALL_ACTION_KEYS:
                if unsafe_key not in ACTION_KEYS or unsafe_key == ex["action_key"]:
                    continue
                estimated_cost = action_cost_usd(ex, unsafe_key)
                route_cost_penalty = cost_penalty(estimated_cost, cost_penalty_weight, cost_reference_usd)
                augmented.append(
                    {
                        **ex,
                        "request_id": f"{ex['request_id']}::synthetic_safety_negative::{unsafe_key}",
                        "action_key": unsafe_key,
                        "action_idx": ACTION_KEYS.index(unsafe_key),
                        "human_reward": negative_reward,
                        "adjusted_reward": clamp_reward(negative_reward - route_cost_penalty),
                        "unsafe_route_penalty": 1.0 - negative_reward,
                        "unsafe_route_penalty_reasons": ["synthetic unsafe route negative example"],
                        "cost_penalty": route_cost_penalty,
                        "estimated_cost_usd": estimated_cost,
                        "avg_rating": -1.0,
                        "feedback_count": 0,
                        "reviewer_count": 0,
                        "positive_count": 0,
                        "negative_count": 1,
                        "source": "synthetic_safety_negative",
                        "synthetic_label": "unsafe_route",
                    }
                )

            if safe_key in ACTION_KEYS:
                estimated_cost = action_cost_usd(ex, safe_key)
                route_cost_penalty = cost_penalty(estimated_cost, cost_penalty_weight, cost_reference_usd)
                augmented.append(
                    {
                        **ex,
                        "request_id": f"{ex['request_id']}::synthetic_safety_positive::{safe_key}",
                        "action_key": safe_key,
                        "action_idx": ACTION_KEYS.index(safe_key),
                        "human_reward": positive_reward,
                        "adjusted_reward": clamp_reward(positive_reward - route_cost_penalty),
                        "unsafe_route_penalty": 0.0,
                        "unsafe_route_penalty_reasons": [],
                        "cost_penalty": route_cost_penalty,
                        "estimated_cost_usd": estimated_cost,
                        "avg_rating": 1.0,
                        "feedback_count": 0,
                        "reviewer_count": 0,
                        "positive_count": 1,
                        "negative_count": 0,
                        "source": "synthetic_safety_positive",
                        "synthetic_label": "safe_route",
                    }
                )

        easy_key = easy_task_preferred_action_key(
            analysis_task_type=str(ex["analysis_task_type"]),
            complexity_level=str(ex["complexity_level"]),
            risk_level=str(ex["risk_level"]),
            source_task_type=str(ex.get("source_task_type", "")),
        )
        if easy_key in ACTION_KEYS:
            estimated_cost = action_cost_usd(ex, easy_key)
            route_cost_penalty = cost_penalty(estimated_cost, cost_penalty_weight, cost_reference_usd)
            augmented.append(
                {
                    **ex,
                    "request_id": f"{ex['request_id']}::synthetic_easy_positive::{easy_key}",
                    "action_key": easy_key,
                    "action_idx": ACTION_KEYS.index(easy_key),
                    "human_reward": easy_positive_reward,
                    "adjusted_reward": clamp_reward(easy_positive_reward - route_cost_penalty),
                    "unsafe_route_penalty": 0.0,
                    "unsafe_route_penalty_reasons": [],
                    "cost_penalty": route_cost_penalty,
                    "estimated_cost_usd": estimated_cost,
                    "avg_rating": 1.0,
                    "feedback_count": 0,
                    "reviewer_count": 0,
                    "positive_count": 1,
                    "negative_count": 0,
                    "source": "synthetic_easy_positive",
                    "synthetic_label": "easy_route",
                }
            )
            for large_key in EASY_TASK_LARGE_ACTION_KEYS:
                if large_key not in ACTION_KEYS or large_key == ex["action_key"]:
                    continue
                estimated_cost = action_cost_usd(ex, large_key)
                route_cost_penalty = cost_penalty(estimated_cost, cost_penalty_weight, cost_reference_usd)
                augmented.append(
                    {
                        **ex,
                        "request_id": f"{ex['request_id']}::synthetic_easy_large_negative::{large_key}",
                        "action_key": large_key,
                        "action_idx": ACTION_KEYS.index(large_key),
                        "human_reward": easy_large_negative_reward,
                        "adjusted_reward": clamp_reward(easy_large_negative_reward - route_cost_penalty),
                        "unsafe_route_penalty": 0.0,
                        "unsafe_route_penalty_reasons": [],
                        "cost_penalty": route_cost_penalty,
                        "estimated_cost_usd": estimated_cost,
                        "avg_rating": -1.0,
                        "feedback_count": 0,
                        "reviewer_count": 0,
                        "positive_count": 0,
                        "negative_count": 1,
                        "source": "synthetic_easy_large_negative",
                        "synthetic_label": "easy_large_route",
                    }
                )

    return augmented


def build_feedback_examples(
    rows: list[dict[str, Any]],
    unsafe_penalty_weight: float = 1.0,
    source_task_types: dict[str, str] | None = None,
    synthetic_safety_examples: bool = True,
    synthetic_negative_reward: float = 0.0,
    synthetic_positive_reward: float = 0.85,
    synthetic_easy_positive_reward: float = 0.9,
    synthetic_easy_large_negative_reward: float = 0.2,
    cost_penalty_weight: float = 0.0,
    cost_reference_usd: float = 0.0007,
) -> list[dict[str, Any]]:
    source_task_types = source_task_types or {}
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        payload = row["payload"]
        if "analysis" not in payload or "strategy" not in payload:
            continue
        analysis = PromptAnalysis.model_validate(payload["analysis"])
        strategy = InferenceStrategy.model_validate(payload["strategy"])
        key = action_key(strategy)
        if key not in ACTION_KEYS or key.startswith("local|"):
            continue
        request_id = payload.get("requestId", "")
        reviewer_id = str(row.get("reviewer_id") or "anonymous")
        if not request_id:
            continue
        training_context = payload.get("trainingContext", {})
        max_completion_tokens = int(
            training_context.get("maxCompletionTokens")
            or payload.get("estimatedCost", {}).get("completionTokens")
            or payload.get("usage", {}).get("completionTokens")
            or 512
        )
        feature_names, features = encode_features(analysis, max_completion_tokens)
        route_penalty, penalty_reasons = unsafe_route_penalty(analysis, strategy)
        weighted_penalty = min(0.65, route_penalty * max(0.0, unsafe_penalty_weight))
        estimated_cost_usd = selected_cost(analysis, strategy, max_completion_tokens).totalCostUsd
        route_cost_penalty = cost_penalty(estimated_cost_usd, cost_penalty_weight, cost_reference_usd)
        source_task_type = source_task_type_for_payload(payload, source_task_types)
        if request_id not in grouped:
            grouped[request_id] = {
                "request_id": payload.get("requestId", ""),
                "analysis": analysis.model_dump(mode="json"),
                "max_completion_tokens": max_completion_tokens,
                "feature_names": feature_names,
                "features": features,
                "action_key": key,
                "action_idx": ACTION_KEYS.index(key),
                "analysis_task_type": analysis.taskType.value,
                "complexity_level": analysis.complexityLevel,
                "risk_level": analysis.riskLevel.value,
                "source_task_type": source_task_type,
                "unsafe_route_penalty": weighted_penalty,
                "unsafe_route_penalty_reasons": penalty_reasons,
                "cost_penalty": route_cost_penalty,
                "estimated_cost_usd": estimated_cost_usd,
                "reviewer_feedback": {},
            }
        feedback_by_reviewer = grouped[request_id]["reviewer_feedback"]
        if reviewer_id in feedback_by_reviewer:
            continue
        feedback_by_reviewer[reviewer_id] = {
            "human_reward": float(row["reward"]),
            "rating": int(row["rating"]),
            "quality_score": row.get("quality_score"),
        }

    examples: list[dict[str, Any]] = []
    for item in grouped.values():
        feedback = list(item["reviewer_feedback"].values())
        if not feedback:
            continue
        rewards = [float(row["human_reward"]) for row in feedback]
        ratings = [int(row["rating"]) for row in feedback]
        human_reward = sum(rewards) / len(rewards)
        unsafe_penalty = float(item["unsafe_route_penalty"])
        route_cost_penalty = float(item.get("cost_penalty", 0.0))
        adjusted_reward = clamp_reward(human_reward - unsafe_penalty - route_cost_penalty)
        examples.append(
            {
                "request_id": item["request_id"],
                "analysis": item["analysis"],
                "max_completion_tokens": item["max_completion_tokens"],
                "feature_names": item["feature_names"],
                "features": item["features"],
                "action_key": item["action_key"],
                "action_idx": item["action_idx"],
                "analysis_task_type": item["analysis_task_type"],
                "complexity_level": item["complexity_level"],
                "risk_level": item["risk_level"],
                "source_task_type": item["source_task_type"],
                "human_reward": human_reward,
                "adjusted_reward": adjusted_reward,
                "unsafe_route_penalty": unsafe_penalty,
                "unsafe_route_penalty_reasons": item["unsafe_route_penalty_reasons"],
                "cost_penalty": route_cost_penalty,
                "estimated_cost_usd": item["estimated_cost_usd"],
                "avg_rating": sum(ratings) / len(ratings),
                "feedback_count": len(feedback),
                "reviewer_count": len(item["reviewer_feedback"]),
                "positive_count": sum(1 for value in ratings if value > 0),
                "negative_count": sum(1 for value in ratings if value < 0),
                "source": "human_feedback",
                "synthetic_label": "",
            }
        )
    if synthetic_safety_examples:
        examples = augment_with_synthetic_routing_examples(
            examples,
            negative_reward=synthetic_negative_reward,
            positive_reward=synthetic_positive_reward,
            easy_positive_reward=synthetic_easy_positive_reward,
            easy_large_negative_reward=synthetic_easy_large_negative_reward,
            cost_penalty_weight=cost_penalty_weight,
            cost_reference_usd=cost_reference_usd,
        )
    return examples


def load_feedback_examples(
    limit: int | None,
    unsafe_penalty_weight: float,
    source_task_types: dict[str, str],
    synthetic_safety_examples: bool,
    synthetic_negative_reward: float,
    synthetic_positive_reward: float,
    synthetic_easy_positive_reward: float,
    synthetic_easy_large_negative_reward: float,
    cost_penalty_weight: float,
    cost_reference_usd: float,
) -> list[dict[str, Any]]:
    store = UsageStore()
    rows = store.feedback_training_rows(limit=limit)
    return build_feedback_examples(
        rows,
        unsafe_penalty_weight=unsafe_penalty_weight,
        source_task_types=source_task_types,
        synthetic_safety_examples=synthetic_safety_examples,
        synthetic_negative_reward=synthetic_negative_reward,
        synthetic_positive_reward=synthetic_positive_reward,
        synthetic_easy_positive_reward=synthetic_easy_positive_reward,
        synthetic_easy_large_negative_reward=synthetic_easy_large_negative_reward,
        cost_penalty_weight=cost_penalty_weight,
        cost_reference_usd=cost_reference_usd,
    )


def load_json_file(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON array.")
    return [dict(item) for item in payload]


def normalize_usage_payload(row: dict[str, Any]) -> dict[str, Any]:
    payload = row.get("payload", row)
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError("usage row payload must be a JSON object.")
    return payload


def load_feedback_examples_from_json(
    feedback_json_path: str,
    usage_json_path: str | None,
    limit: int | None,
    unsafe_penalty_weight: float,
    source_task_types: dict[str, str],
    synthetic_safety_examples: bool,
    synthetic_negative_reward: float,
    synthetic_positive_reward: float,
    synthetic_easy_positive_reward: float,
    synthetic_easy_large_negative_reward: float,
    cost_penalty_weight: float,
    cost_reference_usd: float,
) -> list[dict[str, Any]]:
    feedback_rows = load_json_file(feedback_json_path)
    if limit is not None:
        feedback_rows = feedback_rows[:limit]

    payloads_by_request: dict[str, dict[str, Any]] = {}
    if usage_json_path:
        for row in load_json_file(usage_json_path):
            payload = normalize_usage_payload(row)
            request_id = str(payload.get("requestId") or row.get("request_id") or "")
            if request_id:
                payloads_by_request[request_id] = payload

    rows: list[dict[str, Any]] = []
    missing_payload_count = 0
    for feedback in feedback_rows:
        payload = feedback.get("payload")
        if payload is None:
            request_id = str(feedback.get("request_id") or "")
            payload = payloads_by_request.get(request_id)
        if payload is None:
            missing_payload_count += 1
            continue
        if isinstance(payload, str):
            payload = json.loads(payload)
        rows.append(
            {
                "payload": payload,
                "reviewer_id": feedback.get("reviewer_id", "anonymous"),
                "rating": int(feedback["rating"]),
                "quality_score": feedback.get("quality_score"),
                "reward": float(feedback["reward"]),
                "comment": feedback.get("comment", ""),
                "created_at": feedback.get("created_at", ""),
            }
        )

    if missing_payload_count and not usage_json_path:
        raise ValueError(
            f"{feedback_json_path} has {missing_payload_count} feedback rows without payload. "
            "Export usage_events too and pass --usage_json_path, or export feedback rows with payload included."
        )
    return build_feedback_examples(
        rows,
        unsafe_penalty_weight=unsafe_penalty_weight,
        source_task_types=source_task_types,
        synthetic_safety_examples=synthetic_safety_examples,
        synthetic_negative_reward=synthetic_negative_reward,
        synthetic_positive_reward=synthetic_positive_reward,
        synthetic_easy_positive_reward=synthetic_easy_positive_reward,
        synthetic_easy_large_negative_reward=synthetic_easy_large_negative_reward,
        cost_penalty_weight=cost_penalty_weight,
        cost_reference_usd=cost_reference_usd,
    )


def load_initial_model(path: str | None, feature_names: list[str]) -> LinearRoutingModel:
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            model = LinearRoutingModel.from_json(json.load(f))
        if model.feature_names == feature_names:
            return model
        print(
            json.dumps(
                {
                    "warning": "Initial model feature names do not match current encoder; reinitializing RLHF model.",
                    "initial_model_path": path,
                    "initial_feature_count": len(model.feature_names),
                    "current_feature_count": len(feature_names),
                },
                ensure_ascii=False,
            )
        )
    feature_dim = len(feature_names)
    return LinearRoutingModel(
        feature_names=feature_names,
        action_keys=list(ACTION_KEYS),
        weights=[[0.0 for _ in range(feature_dim)] for _ in ACTION_KEYS],
        bias=[0.0 for _ in ACTION_KEYS],
    )


def train_rlhf_bandit(
    examples: list[dict[str, Any]],
    initial_model_path: str | None,
    epochs: int,
    lr: float,
    temperature: float,
    unsafe_penalty_weight: float,
    cost_penalty_weight: float,
    cost_reference_usd: float,
) -> tuple[LinearRoutingModel, list[dict[str, Any]]]:
    if not examples:
        raise ValueError("No feedback examples found. Run the app and submit Good/Bad feedback first.")

    model = load_initial_model(initial_model_path, examples[0]["feature_names"])

    weights = [[float(v) for v in row] for row in model.weights]
    bias = [float(v) for v in model.bias]
    trainable_indices = [idx for idx, key in enumerate(ACTION_KEYS) if not key.startswith("local|")]
    baseline = sum(float(ex.get("adjusted_reward", ex["human_reward"])) for ex in examples) / max(1, len(examples))
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_reward = 0.0
        total_human_reward = 0.0
        total_penalty = 0.0
        total_cost_penalty = 0.0
        for ex in examples:
            features = ex["features"]
            action_idx = int(ex["action_idx"])
            if action_idx not in trainable_indices:
                continue
            local_action_idx = trainable_indices.index(action_idx)
            scores = [dot(weights[idx], features, bias[idx]) for idx in trainable_indices]
            probs = softmax(scores, temperature)
            reward = float(ex.get("adjusted_reward", ex["human_reward"]))
            advantage = reward - baseline
            chosen_prob = max(probs[local_action_idx], 1e-9)
            total_loss += -math.log(chosen_prob) * advantage

            for local_idx, global_idx in enumerate(trainable_indices):
                grad = (1.0 if local_idx == local_action_idx else 0.0) - probs[local_idx]
                step = lr * advantage * grad
                for feature_idx, value in enumerate(features):
                    weights[global_idx][feature_idx] += step * value
                bias[global_idx] += step

            baseline = 0.95 * baseline + 0.05 * reward
            total_reward += reward
            total_human_reward += float(ex["human_reward"])
            total_penalty += float(ex.get("unsafe_route_penalty", 0.0))
            total_cost_penalty += float(ex.get("cost_penalty", 0.0))

        eval_metrics = evaluate_weights(weights, bias, examples, temperature)
        history.append(
            {
                "epoch": epoch,
                "avg_logged_reward": total_human_reward / max(1, len(examples)),
                "avg_adjusted_reward": total_reward / max(1, len(examples)),
                "avg_unsafe_route_penalty": total_penalty / max(1, len(examples)),
                "avg_cost_penalty": total_cost_penalty / max(1, len(examples)),
                "avg_reviewers_per_example": eval_metrics["avg_reviewers_per_example"],
                "policy_match_rate": eval_metrics["policy_match_rate"],
                "positive_match_rate": eval_metrics["positive_match_rate"],
                "loss": total_loss / max(1, len(examples)),
            }
        )

    trained = LinearRoutingModel(
        feature_names=list(examples[0]["feature_names"]),
        action_keys=list(ACTION_KEYS),
        weights=weights,
        bias=bias,
        model_type="linear_rlhf_bandit_policy",
        metadata={
            "algorithm": "logged_feedback_policy_gradient",
            "feedback_examples": len(examples),
            "feedback_aggregation": "mean_reward_per_request_latest_per_reviewer",
            "epochs": epochs,
            "lr": lr,
            "temperature": temperature,
            "initial_model_path": initial_model_path or "",
            "reward": "adjusted_reward = clamp(human_reward - unsafe_route_penalty - cost_penalty, 0, 1)",
            "unsafe_route_penalty_max": 0.65,
            "unsafe_penalty_weight": unsafe_penalty_weight,
            "cost_penalty_weight": cost_penalty_weight,
            "cost_reference_usd": cost_reference_usd,
            "synthetic_example_count": sum(1 for ex in examples if str(ex.get("source", "")).startswith("synthetic_")),
            "synthetic_safety_negative_count": sum(1 for ex in examples if ex.get("source") == "synthetic_safety_negative"),
            "synthetic_safety_positive_count": sum(1 for ex in examples if ex.get("source") == "synthetic_safety_positive"),
            "synthetic_easy_positive_count": sum(1 for ex in examples if ex.get("source") == "synthetic_easy_positive"),
            "synthetic_easy_large_negative_count": sum(1 for ex in examples if ex.get("source") == "synthetic_easy_large_negative"),
            "source_task_type_count": sum(1 for ex in examples if ex.get("source_task_type")),
        },
    )
    return trained, history


def evaluate_weights(
    weights: list[list[float]],
    bias: list[float],
    examples: list[dict[str, Any]],
    temperature: float,
) -> dict[str, Any]:
    trainable_indices = [idx for idx, key in enumerate(ACTION_KEYS) if not key.startswith("local|")]
    matches = 0
    positive = 0
    positive_matches = 0
    route_counts: dict[str, int] = {}
    reviewer_total = 0
    total_human_reward = 0.0
    total_adjusted_reward = 0.0
    total_penalty = 0.0
    total_cost_penalty = 0.0
    total_estimated_cost = 0.0
    penalized_count = 0
    synthetic_count = 0
    synthetic_negative_count = 0
    synthetic_positive_count = 0
    synthetic_easy_positive_count = 0
    synthetic_easy_large_negative_count = 0
    source_task_type_count = 0

    for ex in examples:
        features = ex["features"]
        scores = [dot(weights[idx], features, bias[idx]) for idx in trainable_indices]
        probs = softmax(scores, temperature)
        best_local = max(range(len(probs)), key=lambda idx: probs[idx])
        pred_idx = trainable_indices[best_local]
        pred_key = ACTION_KEYS[pred_idx]
        route = pred_key.split("|", maxsplit=1)[0]
        route_counts[route] = route_counts.get(route, 0) + 1
        is_match = pred_idx == int(ex["action_idx"])
        matches += int(is_match)
        if float(ex["human_reward"]) >= 0.5:
            positive += 1
            positive_matches += int(is_match)
        reviewer_total += int(ex.get("reviewer_count", 1))
        total_human_reward += float(ex["human_reward"])
        total_adjusted_reward += float(ex.get("adjusted_reward", ex["human_reward"]))
        penalty = float(ex.get("unsafe_route_penalty", 0.0))
        total_penalty += penalty
        total_cost_penalty += float(ex.get("cost_penalty", 0.0))
        total_estimated_cost += float(ex.get("estimated_cost_usd", 0.0))
        penalized_count += int(penalty > 0.0)
        source = str(ex.get("source", ""))
        synthetic_count += int(source.startswith("synthetic_"))
        synthetic_negative_count += int(source == "synthetic_safety_negative")
        synthetic_positive_count += int(source == "synthetic_safety_positive")
        synthetic_easy_positive_count += int(source == "synthetic_easy_positive")
        synthetic_easy_large_negative_count += int(source == "synthetic_easy_large_negative")
        source_task_type_count += int(bool(ex.get("source_task_type")))

    return {
        "num_examples": len(examples),
        "avg_reviewers_per_example": reviewer_total / max(1, len(examples)),
        "avg_human_reward": total_human_reward / max(1, len(examples)),
        "avg_adjusted_reward": total_adjusted_reward / max(1, len(examples)),
        "avg_unsafe_route_penalty": total_penalty / max(1, len(examples)),
        "avg_cost_penalty": total_cost_penalty / max(1, len(examples)),
        "avg_estimated_cost_usd": total_estimated_cost / max(1, len(examples)),
        "penalized_example_count": penalized_count,
        "penalized_example_rate": penalized_count / max(1, len(examples)),
        "synthetic_example_count": synthetic_count,
        "synthetic_example_rate": synthetic_count / max(1, len(examples)),
        "synthetic_safety_negative_count": synthetic_negative_count,
        "synthetic_safety_positive_count": synthetic_positive_count,
        "synthetic_easy_positive_count": synthetic_easy_positive_count,
        "synthetic_easy_large_negative_count": synthetic_easy_large_negative_count,
        "source_task_type_count": source_task_type_count,
        "source_task_type_rate": source_task_type_count / max(1, len(examples)),
        "policy_match_rate": matches / max(1, len(examples)),
        "positive_match_rate": positive_matches / max(1, positive),
        "route_counts": route_counts,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the routing policy from human feedback rewards.")
    parser.add_argument("--sqlite_path", default="outputs/usage.db")
    parser.add_argument("--feedback_json_path", default="")
    parser.add_argument("--usage_json_path", default="")
    parser.add_argument("--initial_model_path", default="outputs/rl_routing_policy.json")
    parser.add_argument("--output_path", default="outputs/rlhf_routing_policy.json")
    parser.add_argument("--metrics_path", default="outputs/rlhf_routing_policy_metrics.json")
    parser.add_argument("--history_path", default="outputs/rlhf_routing_policy_history.json")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--unsafe_penalty_weight", type=float, default=1.0)
    parser.add_argument(
        "--source_task_paths",
        nargs="*",
        default=["data/service_request_suite.json", "data/service_eval_questions.json"],
    )
    parser.add_argument("--disable_synthetic_safety_examples", action="store_true")
    parser.add_argument("--synthetic_negative_reward", type=float, default=0.0)
    parser.add_argument("--synthetic_positive_reward", type=float, default=0.85)
    parser.add_argument("--synthetic_easy_positive_reward", type=float, default=0.9)
    parser.add_argument("--synthetic_easy_large_negative_reward", type=float, default=0.2)
    parser.add_argument("--cost_penalty_weight", type=float, default=0.2)
    parser.add_argument("--cost_reference_usd", type=float, default=0.0007)
    args = parser.parse_args()

    if args.sqlite_path:
        os.environ["SQLITE_PATH"] = args.sqlite_path

    source_task_types = load_source_task_types(args.source_task_paths)

    if args.feedback_json_path:
        examples = load_feedback_examples_from_json(
            feedback_json_path=args.feedback_json_path,
            usage_json_path=args.usage_json_path or None,
            limit=args.limit or None,
            unsafe_penalty_weight=args.unsafe_penalty_weight,
            source_task_types=source_task_types,
            synthetic_safety_examples=not args.disable_synthetic_safety_examples,
            synthetic_negative_reward=args.synthetic_negative_reward,
            synthetic_positive_reward=args.synthetic_positive_reward,
            synthetic_easy_positive_reward=args.synthetic_easy_positive_reward,
            synthetic_easy_large_negative_reward=args.synthetic_easy_large_negative_reward,
            cost_penalty_weight=args.cost_penalty_weight,
            cost_reference_usd=args.cost_reference_usd,
        )
    else:
        examples = load_feedback_examples(
            limit=args.limit or None,
            unsafe_penalty_weight=args.unsafe_penalty_weight,
            source_task_types=source_task_types,
            synthetic_safety_examples=not args.disable_synthetic_safety_examples,
            synthetic_negative_reward=args.synthetic_negative_reward,
            synthetic_positive_reward=args.synthetic_positive_reward,
            synthetic_easy_positive_reward=args.synthetic_easy_positive_reward,
            synthetic_easy_large_negative_reward=args.synthetic_easy_large_negative_reward,
            cost_penalty_weight=args.cost_penalty_weight,
            cost_reference_usd=args.cost_reference_usd,
        )
    model, history = train_rlhf_bandit(
        examples=examples,
        initial_model_path=args.initial_model_path,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        unsafe_penalty_weight=args.unsafe_penalty_weight,
        cost_penalty_weight=args.cost_penalty_weight,
        cost_reference_usd=args.cost_reference_usd,
    )
    metrics = evaluate_weights(model.weights, model.bias, examples, args.temperature)

    for path in [args.output_path, args.metrics_path, args.history_path]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, ensure_ascii=False, indent=2)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(args.history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print(json.dumps({"model_path": args.output_path, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
