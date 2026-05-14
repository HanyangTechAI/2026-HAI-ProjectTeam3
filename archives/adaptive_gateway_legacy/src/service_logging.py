import json
import os
import time
from typing import Any, Dict, List


class JsonlRequestStore:
    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> Dict[str, Any]:
        record = dict(record)
        record.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_recent(self, limit: int = 50) -> List[Dict[str, Any]]:
        if not os.path.exists(self.path):
            return []

        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        rows = []
        for line in lines[-max(1, limit):]:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    def metrics(self) -> Dict[str, Any]:
        all_rows = self.read_recent(limit=100000)
        feedback_rows = [row for row in all_rows if row.get("type") == "feedback"]
        rows = [row for row in all_rows if row.get("type") != "feedback"]
        total = len(rows)
        if total == 0:
            return {
                "total_requests": 0,
                "feedback_count": len(feedback_rows),
                "labeled_requests": 0,
                "accuracy": None,
                "avg_reward": None,
                "avg_total_tokens": None,
                "estimated_cost_units": 0.0,
                "fixed_large_cost_units": 0.0,
                "estimated_savings_units": 0.0,
                "estimated_savings_rate": None,
                "action_hist": {},
                "route_hist": {},
                "task_type_hist": {},
                "policy_source_hist": {},
            }

        labeled = [row for row in rows if row.get("correct") is not None]
        correct = sum(1 for row in labeled if row.get("correct"))
        reward_values = [
            row["reward_breakdown"]["total_reward"]
            for row in rows
            if isinstance(row.get("reward_breakdown"), dict)
        ]
        token_values = [int(row.get("total_tokens", 0)) for row in rows]
        action_hist: Dict[str, int] = {}
        route_hist: Dict[str, int] = {}
        task_type_hist: Dict[str, int] = {}
        policy_source_hist: Dict[str, int] = {}
        estimated_cost_units = 0.0
        fixed_large_cost_units = 0.0

        for row in rows:
            action_key = str(row.get("chosen_action_idx", "unknown"))
            action_hist[action_key] = action_hist.get(action_key, 0) + 1

            action = row.get("chosen_action") or {}
            route = str(action.get("model_route", "unknown"))
            route_hist[route] = route_hist.get(route, 0) + 1

            task = row.get("task") or {}
            task_type = str(task.get("task_type", "unknown"))
            task_type_hist[task_type] = task_type_hist.get(task_type, 0) + 1

            policy = row.get("policy") or {}
            policy_source = str(policy.get("policy_source", "unknown"))
            policy_source_hist[policy_source] = policy_source_hist.get(policy_source, 0) + 1

            multiplier = 2.0 if route == "large" else 1.0
            total_tokens = float(row.get("total_tokens", 0))
            estimated_cost_units += total_tokens * multiplier
            fixed_large_cost_units += total_tokens * 2.0

        estimated_savings_units = fixed_large_cost_units - estimated_cost_units

        return {
            "total_requests": total,
            "feedback_count": len(feedback_rows),
            "labeled_requests": len(labeled),
            "accuracy": correct / len(labeled) if labeled else None,
            "avg_reward": sum(reward_values) / len(reward_values) if reward_values else None,
            "avg_total_tokens": sum(token_values) / total,
            "estimated_cost_units": estimated_cost_units,
            "fixed_large_cost_units": fixed_large_cost_units,
            "estimated_savings_units": estimated_savings_units,
            "estimated_savings_rate": (
                estimated_savings_units / fixed_large_cost_units
                if fixed_large_cost_units > 0
                else None
            ),
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
            "route_hist": dict(sorted(route_hist.items(), key=lambda x: x[1], reverse=True)),
            "task_type_hist": dict(sorted(task_type_hist.items(), key=lambda x: x[1], reverse=True)),
            "policy_source_hist": dict(sorted(policy_source_hist.items(), key=lambda x: x[1], reverse=True)),
        }
