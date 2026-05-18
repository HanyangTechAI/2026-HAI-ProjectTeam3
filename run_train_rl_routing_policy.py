import argparse
import json
import math
import os
import random
from typing import Any

from backend.app.analyzer import analyze_prompt
from backend.app.model_policy import (
    ACTION_KEYS,
    LinearRoutingModel,
    encode_features,
    strategy_from_action_key,
)
from backend.app.policy import selected_cost
from backend.app.schemas import InferenceStrategy, ModelRoute, PromptAnalysis, ReasoningDepth, RetryStrategy, TaskType, RiskLevel


def load_suite(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [row for row in rows if row.get("request")]


def build_examples(rows: list[dict[str, Any]], max_completion_values: list[int]) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        prompt = str(row["request"])
        analysis = analyze_prompt(prompt)
        for max_tokens in max_completion_values:
            feature_names, features = encode_features(analysis, max_tokens)
            examples.append(
                {
                    "prompt": prompt,
                    "task_type": row.get("task_type", analysis.taskType.value),
                    "max_completion_tokens": max_tokens,
                    "feature_names": feature_names,
                    "features": features,
                    "analysis": analysis,
                }
            )
    return examples


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def expected_quality(analysis: PromptAnalysis, strategy: InferenceStrategy) -> float:
    route = strategy.modelRoute
    depth = strategy.reasoningDepth
    task = analysis.taskType

    if route == ModelRoute.LOCAL:
        score = 0.10
    elif route == ModelRoute.OPENAI_LARGE:
        score = 0.88
    elif route == ModelRoute.GEMINI_LARGE:
        score = 0.84
    elif route == ModelRoute.OPENAI_SMALL:
        score = 0.74
    else:
        score = 0.70

    if task == TaskType.WRITING:
        score += 0.12 if route == ModelRoute.OPENAI_SMALL else 0.02 if route == ModelRoute.OPENAI_LARGE else -0.02
    elif task in {TaskType.SUMMARIZATION, TaskType.CLASSIFICATION} and analysis.riskLevel == RiskLevel.LOW:
        score += 0.10 if route == ModelRoute.GEMINI_SMALL else -0.01 if route.value.endswith("large") else 0.02
    elif task == TaskType.MATH:
        score += 0.08 if route == ModelRoute.GEMINI_LARGE else 0.04 if route == ModelRoute.OPENAI_LARGE else -0.08
    elif task == TaskType.CODING:
        score += 0.10 if route == ModelRoute.OPENAI_LARGE else -0.08 if route.value.endswith("small") else 0.03
    elif task == TaskType.STOCK:
        score += 0.12 if route == ModelRoute.OPENAI_LARGE else 0.06 if route == ModelRoute.GEMINI_LARGE else -0.12
        score += 0.05 if strategy.verify else -0.08
        score += 0.03 if depth == ReasoningDepth.LONG else -0.05

    if analysis.riskLevel == RiskLevel.HIGH:
        score += 0.09 if route.value.endswith("large") else -0.14
        score += 0.05 if strategy.verify else -0.10
        score += 0.02 if strategy.retry == RetryStrategy.ONCE else -0.03

    if analysis.reasoningNeed == ReasoningDepth.LONG:
        score += 0.06 if depth == ReasoningDepth.LONG else -0.12 if depth == ReasoningDepth.NONE else -0.03
    elif analysis.reasoningNeed == ReasoningDepth.SHORT:
        score += 0.03 if depth == ReasoningDepth.SHORT else -0.04 if depth == ReasoningDepth.NONE else -0.01
    elif depth == ReasoningDepth.LONG:
        score -= 0.03

    if analysis.promptTokensEstimate > 1200:
        score += 0.04 if strategy.contextCompression else -0.03
    elif strategy.contextCompression:
        score -= 0.03

    return clamp(score)


def latency_penalty(strategy: InferenceStrategy) -> float:
    route_penalty = {
        ModelRoute.LOCAL: 0.00,
        ModelRoute.GEMINI_SMALL: 0.05,
        ModelRoute.OPENAI_SMALL: 0.06,
        ModelRoute.GEMINI_LARGE: 0.14,
        ModelRoute.OPENAI_LARGE: 0.16,
    }[strategy.modelRoute]
    depth_penalty = {
        ReasoningDepth.NONE: 0.00,
        ReasoningDepth.SHORT: 0.03,
        ReasoningDepth.LONG: 0.09,
    }[strategy.reasoningDepth]
    verify_penalty = 0.04 if strategy.verify else 0.00
    retry_penalty = 0.04 if strategy.retry == RetryStrategy.ONCE else 0.00
    return route_penalty + depth_penalty + verify_penalty + retry_penalty


def reward_for_action(
    analysis: PromptAnalysis,
    max_completion_tokens: int,
    action_key: str,
    cost_weight: float,
    latency_weight: float,
) -> dict[str, float]:
    strategy = strategy_from_action_key(action_key, reason="")
    cost = selected_cost(analysis, strategy, max_completion_tokens).totalCostUsd
    action_costs = [
        selected_cost(analysis, strategy_from_action_key(key, reason=""), max_completion_tokens).totalCostUsd
        for key in ACTION_KEYS
        if not key.startswith("local|")
    ]
    max_cost = max(action_costs) if action_costs else 1.0
    quality = expected_quality(analysis, strategy)
    cost_norm = cost / max(max_cost, 1e-9)
    latency = latency_penalty(strategy)
    total = quality - cost_weight * cost_norm - latency_weight * latency
    return {
        "total": total,
        "quality": quality,
        "cost_norm": cost_norm,
        "latency": latency,
        "cost_usd": cost,
    }


def dot(row: list[float], features: list[float], bias: float) -> float:
    return sum(w * x for w, x in zip(row, features)) + bias


def softmax(scores: list[float], temperature: float) -> list[float]:
    scaled = [score / max(temperature, 1e-6) for score in scores]
    peak = max(scaled)
    exp_values = [math.exp(score - peak) for score in scaled]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def sample_index(probs: list[float], rng: random.Random) -> int:
    threshold = rng.random()
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if threshold <= cumulative:
            return idx
    return len(probs) - 1


def train_bandit(
    examples: list[dict[str, Any]],
    epochs: int,
    lr: float,
    temperature: float,
    cost_weight: float,
    latency_weight: float,
    seed: int,
) -> tuple[LinearRoutingModel, list[dict[str, Any]]]:
    if not examples:
        raise ValueError("No training examples were generated.")

    rng = random.Random(seed)
    feature_names = list(examples[0]["feature_names"])
    feature_dim = len(feature_names)
    weights = [[0.0 for _ in range(feature_dim)] for _ in ACTION_KEYS]
    bias = [0.0 for _ in ACTION_KEYS]
    trainable_indices = [idx for idx, key in enumerate(ACTION_KEYS) if not key.startswith("local|")]
    baseline = 0.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        rng.shuffle(examples)
        total_reward = 0.0
        chosen_counts: dict[str, int] = {}
        for ex in examples:
            features = ex["features"]
            scores = [dot(weights[idx], features, bias[idx]) for idx in trainable_indices]
            probs = softmax(scores, temperature)
            local_choice = sample_index(probs, rng)
            action_idx = trainable_indices[local_choice]
            reward = reward_for_action(
                analysis=ex["analysis"],
                max_completion_tokens=int(ex["max_completion_tokens"]),
                action_key=ACTION_KEYS[action_idx],
                cost_weight=cost_weight,
                latency_weight=latency_weight,
            )["total"]
            baseline = 0.95 * baseline + 0.05 * reward
            advantage = reward - baseline

            for local_idx, global_idx in enumerate(trainable_indices):
                grad = (1.0 if local_idx == local_choice else 0.0) - probs[local_idx]
                step = lr * advantage * grad
                for feature_idx, value in enumerate(features):
                    weights[global_idx][feature_idx] += step * value
                bias[global_idx] += step

            total_reward += reward
            chosen_counts[ACTION_KEYS[action_idx]] = chosen_counts.get(ACTION_KEYS[action_idx], 0) + 1

        eval_metrics = evaluate_weights(weights, bias, examples, cost_weight, latency_weight)
        history.append(
            {
                "epoch": epoch,
                "sampled_avg_reward": total_reward / max(1, len(examples)),
                "greedy_avg_reward": eval_metrics["avg_reward"],
                "greedy_avg_quality": eval_metrics["avg_quality"],
                "greedy_avg_cost_usd": eval_metrics["avg_cost_usd"],
                "chosen_counts": chosen_counts,
            }
        )

    model = LinearRoutingModel(
        feature_names=feature_names,
        action_keys=list(ACTION_KEYS),
        weights=weights,
        bias=bias,
        model_type="linear_contextual_bandit_policy",
        metadata={
            "algorithm": "reinforce_contextual_bandit",
            "reward": {
                "quality": "heuristic expected answer quality",
                "cost_weight": cost_weight,
                "latency_weight": latency_weight,
            },
            "epochs": epochs,
            "lr": lr,
            "temperature": temperature,
            "seed": seed,
        },
    )
    return model, history


def evaluate_weights(
    weights: list[list[float]],
    bias: list[float],
    examples: list[dict[str, Any]],
    cost_weight: float,
    latency_weight: float,
) -> dict[str, Any]:
    route_counts: dict[str, int] = {}
    task_counts: dict[str, int] = {}
    total_reward = 0.0
    total_quality = 0.0
    total_cost = 0.0
    trainable_indices = [idx for idx, key in enumerate(ACTION_KEYS) if not key.startswith("local|")]

    for ex in examples:
        features = ex["features"]
        best_idx = max(trainable_indices, key=lambda idx: dot(weights[idx], features, bias[idx]))
        action = ACTION_KEYS[best_idx]
        parts = reward_for_action(
            analysis=ex["analysis"],
            max_completion_tokens=int(ex["max_completion_tokens"]),
            action_key=action,
            cost_weight=cost_weight,
            latency_weight=latency_weight,
        )
        route = action.split("|", maxsplit=1)[0]
        task = str(ex["task_type"])
        route_counts[route] = route_counts.get(route, 0) + 1
        task_counts[task] = task_counts.get(task, 0) + 1
        total_reward += parts["total"]
        total_quality += parts["quality"]
        total_cost += parts["cost_usd"]

    n = max(1, len(examples))
    return {
        "num_examples": len(examples),
        "avg_reward": total_reward / n,
        "avg_quality": total_quality / n,
        "avg_cost_usd": total_cost / n,
        "route_counts": route_counts,
        "task_counts": task_counts,
    }


def evaluate(model: LinearRoutingModel, examples: list[dict[str, Any]], cost_weight: float, latency_weight: float) -> dict[str, Any]:
    return evaluate_weights(model.weights, model.bias, examples, cost_weight, latency_weight)


def main():
    parser = argparse.ArgumentParser(description="Train an RL-style contextual bandit routing policy for the AI gateway.")
    parser.add_argument("--suite_path", default="data/service_request_suite.json")
    parser.add_argument("--output_path", default="outputs/rl_routing_policy.json")
    parser.add_argument("--metrics_path", default="outputs/rl_routing_policy_metrics.json")
    parser.add_argument("--history_path", default="outputs/rl_routing_policy_history.json")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.08)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cost_weight", type=float, default=0.18)
    parser.add_argument("--latency_weight", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max_completion_values", default="128,256,512,1024")
    args = parser.parse_args()

    max_values = [int(x.strip()) for x in args.max_completion_values.split(",") if x.strip()]
    rows = load_suite(args.suite_path)
    examples = build_examples(rows, max_values)
    model, history = train_bandit(
        examples=examples,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        cost_weight=args.cost_weight,
        latency_weight=args.latency_weight,
        seed=args.seed,
    )
    metrics = evaluate(model, examples, args.cost_weight, args.latency_weight)

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
