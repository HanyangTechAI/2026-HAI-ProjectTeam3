import argparse
import json
import math
import os
from typing import Any

from backend.app.model_policy import ACTION_KEYS, LinearRoutingModel, action_key, encode_features
from backend.app.schemas import InferenceStrategy, ModelRoute, PromptAnalysis, RiskLevel, TaskType
from backend.app.store import UsageStore


def softmax(scores: list[float], temperature: float) -> list[float]:
    scaled = [score / max(temperature, 1e-6) for score in scores]
    peak = max(scaled)
    values = [math.exp(score - peak) for score in scaled]
    total = sum(values)
    return [value / total for value in values]


def dot(row: list[float], features: list[float], bias: float) -> float:
    return sum(w * x for w, x in zip(row, features)) + bias


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


def build_feedback_examples(rows: list[dict[str, Any]], unsafe_penalty_weight: float = 1.0) -> list[dict[str, Any]]:
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
        if request_id not in grouped:
            grouped[request_id] = {
                "request_id": payload.get("requestId", ""),
                "feature_names": feature_names,
                "features": features,
                "action_key": key,
                "action_idx": ACTION_KEYS.index(key),
                "unsafe_route_penalty": weighted_penalty,
                "unsafe_route_penalty_reasons": penalty_reasons,
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
        adjusted_reward = max(0.0, min(1.0, human_reward - unsafe_penalty))
        examples.append(
            {
                "request_id": item["request_id"],
                "feature_names": item["feature_names"],
                "features": item["features"],
                "action_key": item["action_key"],
                "action_idx": item["action_idx"],
                "human_reward": human_reward,
                "adjusted_reward": adjusted_reward,
                "unsafe_route_penalty": unsafe_penalty,
                "unsafe_route_penalty_reasons": item["unsafe_route_penalty_reasons"],
                "avg_rating": sum(ratings) / len(ratings),
                "feedback_count": len(feedback),
                "reviewer_count": len(item["reviewer_feedback"]),
                "positive_count": sum(1 for value in ratings if value > 0),
                "negative_count": sum(1 for value in ratings if value < 0),
            }
        )
    return examples


def load_feedback_examples(limit: int | None, unsafe_penalty_weight: float) -> list[dict[str, Any]]:
    store = UsageStore()
    rows = store.feedback_training_rows(limit=limit)
    return build_feedback_examples(rows, unsafe_penalty_weight=unsafe_penalty_weight)


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
    return build_feedback_examples(rows, unsafe_penalty_weight=unsafe_penalty_weight)


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

        eval_metrics = evaluate_weights(weights, bias, examples, temperature)
        history.append(
            {
                "epoch": epoch,
                "avg_logged_reward": total_human_reward / max(1, len(examples)),
                "avg_adjusted_reward": total_reward / max(1, len(examples)),
                "avg_unsafe_route_penalty": total_penalty / max(1, len(examples)),
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
            "reward": "adjusted_reward = clamp(human_reward - unsafe_route_penalty, 0, 1)",
            "unsafe_route_penalty_max": 0.65,
            "unsafe_penalty_weight": unsafe_penalty_weight,
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
    penalized_count = 0

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
        penalized_count += int(penalty > 0.0)

    return {
        "num_examples": len(examples),
        "avg_reviewers_per_example": reviewer_total / max(1, len(examples)),
        "avg_human_reward": total_human_reward / max(1, len(examples)),
        "avg_adjusted_reward": total_adjusted_reward / max(1, len(examples)),
        "avg_unsafe_route_penalty": total_penalty / max(1, len(examples)),
        "penalized_example_count": penalized_count,
        "penalized_example_rate": penalized_count / max(1, len(examples)),
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
    args = parser.parse_args()

    if args.sqlite_path:
        os.environ["SQLITE_PATH"] = args.sqlite_path

    if args.feedback_json_path:
        examples = load_feedback_examples_from_json(
            feedback_json_path=args.feedback_json_path,
            usage_json_path=args.usage_json_path or None,
            limit=args.limit or None,
            unsafe_penalty_weight=args.unsafe_penalty_weight,
        )
    else:
        examples = load_feedback_examples(
            limit=args.limit or None,
            unsafe_penalty_weight=args.unsafe_penalty_weight,
        )
    model, history = train_rlhf_bandit(
        examples=examples,
        initial_model_path=args.initial_model_path,
        epochs=args.epochs,
        lr=args.lr,
        temperature=args.temperature,
        unsafe_penalty_weight=args.unsafe_penalty_weight,
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
