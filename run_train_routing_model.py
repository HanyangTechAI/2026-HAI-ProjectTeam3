import argparse
import json
import os
from typing import Any

from backend.app.analyzer import analyze_prompt
from backend.app.model_policy import ACTION_KEYS, LinearRoutingModel, action_key, encode_features
from backend.app.policy import RuleBasedPolicy


def load_suite(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return [row for row in rows if row.get("request")]


def build_teacher_examples(rows: list[dict[str, Any]], max_completion_values: list[int]):
    teacher = RuleBasedPolicy()
    examples = []
    for row in rows:
        prompt = str(row["request"])
        for max_tokens in max_completion_values:
            analysis = analyze_prompt(prompt)
            strategy, _ = teacher.choose(analysis, max_tokens, force_mock=False)
            key = action_key(strategy)
            if key not in ACTION_KEYS:
                continue
            feature_names, features = encode_features(analysis, max_tokens)
            examples.append(
                {
                    "prompt": prompt,
                    "task_type": row.get("task_type", analysis.taskType.value),
                    "max_completion_tokens": max_tokens,
                    "feature_names": feature_names,
                    "features": features,
                    "label_key": key,
                    "label_idx": ACTION_KEYS.index(key),
                }
            )
    return examples


def train_perceptron(examples: list[dict[str, Any]], epochs: int, lr: float) -> LinearRoutingModel:
    if not examples:
        raise ValueError("No training examples were generated.")

    feature_names = list(examples[0]["feature_names"])
    feature_dim = len(feature_names)
    weights = [[0.0 for _ in range(feature_dim)] for _ in ACTION_KEYS]
    bias = [0.0 for _ in ACTION_KEYS]

    for _ in range(epochs):
        for ex in examples:
            features = ex["features"]
            gold = int(ex["label_idx"])
            scores = [
                sum(w * x for w, x in zip(row, features)) + bias[idx]
                for idx, row in enumerate(weights)
            ]
            pred = max(range(len(scores)), key=lambda idx: scores[idx])
            if pred == gold:
                continue
            for j, value in enumerate(features):
                weights[gold][j] += lr * value
                weights[pred][j] -= lr * value
            bias[gold] += lr
            bias[pred] -= lr

    return LinearRoutingModel(
        feature_names=feature_names,
        action_keys=list(ACTION_KEYS),
        weights=weights,
        bias=bias,
    )


def evaluate(model: LinearRoutingModel, examples: list[dict[str, Any]]) -> dict[str, Any]:
    correct = 0
    task_counts: dict[str, int] = {}
    task_correct: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}

    for ex in examples:
        pred_idx = model.predict_index(ex["features"])
        gold_key = ex["label_key"]
        pred_key = model.action_keys[pred_idx]
        task = str(ex["task_type"])
        correct += int(pred_key == gold_key)
        task_counts[task] = task_counts.get(task, 0) + 1
        task_correct[task] = task_correct.get(task, 0) + int(pred_key == gold_key)
        confusion.setdefault(gold_key, {})
        confusion[gold_key][pred_key] = confusion[gold_key].get(pred_key, 0) + 1

    return {
        "num_examples": len(examples),
        "accuracy": correct / max(1, len(examples)),
        "task_accuracy": {
            task: task_correct.get(task, 0) / count
            for task, count in sorted(task_counts.items())
        },
        "confusion": confusion,
    }


def main():
    parser = argparse.ArgumentParser(description="Train the first learned routing model for the AI gateway.")
    parser.add_argument("--suite_path", default="data/service_request_suite.json")
    parser.add_argument("--output_path", default="outputs/routing_policy.json")
    parser.add_argument("--metrics_path", default="outputs/routing_policy_metrics.json")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--max_completion_values", default="128,256,512,1024")
    args = parser.parse_args()

    max_values = [int(x.strip()) for x in args.max_completion_values.split(",") if x.strip()]
    rows = load_suite(args.suite_path)
    examples = build_teacher_examples(rows, max_values)
    model = train_perceptron(examples, epochs=args.epochs, lr=args.lr)
    metrics = evaluate(model, examples)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, ensure_ascii=False, indent=2)
    with open(args.metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps({"model_path": args.output_path, "metrics": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
