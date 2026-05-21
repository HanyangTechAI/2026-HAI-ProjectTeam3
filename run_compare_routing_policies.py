import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Any

from backend.app.analyzer import analyze_prompt
from backend.app.model_policy import load_model_policy
from backend.app.policy import Policy, RuleBasedPolicy, selected_cost


def load_eval_prompts(paths: list[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"{path} must contain a JSON array.")
        for idx, item in enumerate(payload):
            prompt = item.get("request") or item.get("question") or item.get("prompt")
            if not prompt:
                continue
            key = str(prompt)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "source": path,
                    "source_index": idx,
                    "task_type": item.get("task_type") or item.get("category") or "",
                    "prompt": key,
                    "gold": item.get("gold", ""),
                    "good_criteria": item.get("good_criteria", item.get("note", "")),
                }
            )
    return records


def load_policies(policy_paths: dict[str, str]) -> dict[str, Policy]:
    policies: dict[str, Policy] = {"rule": RuleBasedPolicy()}
    for name, path in policy_paths.items():
        if not path:
            continue
        if not os.path.exists(path):
            print(json.dumps({"warning": "policy file not found", "name": name, "path": path}))
            continue
        policies[name] = load_model_policy(path)
    return policies


def summarize(records: list[dict[str, Any]], policy_names: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"num_prompts": len(records), "policies": {}}
    for name in policy_names:
        route_counts: Counter[str] = Counter()
        task_route_counts: dict[str, Counter[str]] = defaultdict(Counter)
        total_cost = 0.0
        verify_count = 0
        compression_count = 0
        retry_count = 0
        for record in records:
            decision = record["decisions"][name]
            route_counts[decision["modelRoute"]] += 1
            task_route_counts[record["task_type"] or "unknown"][decision["modelRoute"]] += 1
            total_cost += float(decision["estimatedCostUsd"])
            verify_count += int(bool(decision["verify"]))
            compression_count += int(bool(decision["contextCompression"]))
            retry_count += int(decision["retry"] != "none")
        summary["policies"][name] = {
            "routeCounts": dict(route_counts),
            "avgEstimatedCostUsd": total_cost / max(1, len(records)),
            "totalEstimatedCostUsd": total_cost,
            "verifyRate": verify_count / max(1, len(records)),
            "compressionRate": compression_count / max(1, len(records)),
            "retryRate": retry_count / max(1, len(records)),
            "taskRouteCounts": {task: dict(counts) for task, counts in sorted(task_route_counts.items())},
        }

    base = policy_names[0] if policy_names else ""
    if base:
        for name in policy_names[1:]:
            matches = sum(
                1
                for record in records
                if record["decisions"][base]["actionKey"] == record["decisions"][name]["actionKey"]
            )
            summary["policies"][name]["matchRateVs" + base.title()] = matches / max(1, len(records))
    return summary


def compare_policies(
    eval_paths: list[str],
    policy_paths: dict[str, str],
    max_completion_tokens: int,
) -> dict[str, Any]:
    prompts = load_eval_prompts(eval_paths)
    policies = load_policies(policy_paths)
    policy_names = list(policies.keys())
    records: list[dict[str, Any]] = []

    for item in prompts:
        analysis = analyze_prompt(item["prompt"])
        decisions: dict[str, Any] = {}
        for name, policy in policies.items():
            strategy, _ = policy.choose(analysis, max_completion_tokens, force_mock=False)
            cost = selected_cost(analysis, strategy, max_completion_tokens)
            action_key = "|".join(
                [
                    strategy.modelRoute.value,
                    strategy.reasoningDepth.value,
                    str(strategy.verify).lower(),
                    strategy.retry.value,
                    str(strategy.contextCompression).lower(),
                ]
            )
            decisions[name] = {
                "actionKey": action_key,
                "modelRoute": strategy.modelRoute.value,
                "reasoningDepth": strategy.reasoningDepth.value,
                "verify": strategy.verify,
                "retry": strategy.retry.value,
                "contextCompression": strategy.contextCompression,
                "estimatedCostUsd": cost.totalCostUsd,
                "decisionReason": strategy.decisionReason,
            }
        records.append(
            {
                **item,
                "analysis": analysis.model_dump(mode="json"),
                "decisions": decisions,
            }
        )

    return {
        "summary": summarize(records, policy_names),
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare routing policy decisions on offline prompt sets.")
    parser.add_argument(
        "--eval_paths",
        nargs="+",
        default=[
            "data/service_request_suite.json",
            "data/service_eval_questions.json",
        ],
    )
    parser.add_argument("--routing_policy_path", default="outputs/routing_policy.json")
    parser.add_argument("--rl_policy_path", default="outputs/rl_routing_policy.json")
    parser.add_argument("--rlhf_policy_path", default="outputs/rlhf_routing_policy.json")
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    parser.add_argument("--output_path", default="outputs/routing_policy_comparison.json")
    parser.add_argument("--summary_path", default="outputs/routing_policy_comparison_summary.json")
    args = parser.parse_args()

    result = compare_policies(
        eval_paths=args.eval_paths,
        policy_paths={
            "supervised": args.routing_policy_path,
            "rl": args.rl_policy_path,
            "rlhf": args.rlhf_policy_path,
        },
        max_completion_tokens=args.max_completion_tokens,
    )

    for path in [args.output_path, args.summary_path]:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    with open(args.summary_path, "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
