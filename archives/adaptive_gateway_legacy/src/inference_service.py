import os
import random
import uuid
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import extract_gold_answer
from src.demo_samples import load_demo_dataset
from src.execution.task_classifier import task_metadata
from src.llm_client import build_llm_client
from src.policy_utils import (
    compute_action_scores,
    compute_task_aware_heuristic_action_scores,
    estimate_difficulty,
    load_preference_model,
    resolve_policy_checkpoint,
)
from src.service_logging import JsonlRequestStore


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def describe_action(action_desc: dict) -> str:
    verify = "verify" if action_desc["verify"] else "no-verify"
    return f"{action_desc['reasoning_budget']} / {action_desc['model_route']} / {verify}"


@dataclass
class ServiceConfig:
    api_mode: str = "auto"
    embedding_model: str = "hashing:384"
    policy_source: str = "auto"
    checkpoint: str = ""
    topk: int = 3
    log_path: str = "outputs/service_requests.jsonl"


class InferencePolicyService:
    def __init__(self, service_config: ServiceConfig | None = None):
        self.cfg = TrainConfig()
        self.service_config = service_config or ServiceConfig()
        set_seed(self.cfg.seed)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.requested_api_mode = self.service_config.api_mode
        self.resolved_api_mode = resolve_api_mode(self.service_config.api_mode)
        self.action_space = InferenceActionSpace()
        self.request_store = JsonlRequestStore(self.service_config.log_path)
        self.model_path = "heuristic_fallback_policy"
        self.model = None
        self._load_policy()

        llm_client = build_llm_client(self.resolved_api_mode)
        self.controller = AdaptiveInferenceController(
            llm_client=llm_client,
            action_space=self.action_space,
        )

    def _load_policy(self):
        policy_source = self.service_config.policy_source
        if (
            policy_source == "auto"
            and self.resolved_api_mode == "mock"
            and not self.service_config.checkpoint
        ):
            policy_source = "heuristic"

        if policy_source == "heuristic":
            self.model_path = "heuristic_fallback_policy"
            self.model = None
            return

        try:
            self.model_path = resolve_policy_checkpoint(
                self.cfg.output_dir,
                self.service_config.checkpoint or None,
            )
            self.model, _ = load_preference_model(self.model_path, device=self.device)
        except FileNotFoundError:
            if policy_source == "checkpoint" or self.service_config.checkpoint:
                raise
            self.model_path = "heuristic_fallback_policy"
            self.model = None

    def predict(
        self,
        question: str,
        gold: str = "",
        topk: int | None = None,
        log_request: bool = True,
    ) -> Dict[str, Any]:
        question = question.strip()
        gold = gold.strip()
        if not question:
            raise ValueError("question is required")
        task_info = task_metadata(question)

        full_state = build_full_state(
            question=question,
            model_name=self.service_config.embedding_model,
            normalize_embedding=True,
        )
        state_features = full_state["handcrafted"]
        state_embedding = full_state["embedding"]
        difficulty = estimate_difficulty(state_features)

        with torch.no_grad():
            if self.model is None or task_info["task_type"] != "math":
                scores = compute_task_aware_heuristic_action_scores(
                    state_features=state_features,
                    action_space=self.action_space,
                    device=self.device,
                    task_type=task_info["task_type"],
                    text=question,
                )
            else:
                scores = compute_action_scores(
                    model=self.model,
                    state_features=state_features,
                    state_embedding=state_embedding,
                    action_space=self.action_space,
                    device=self.device,
                )

        sorted_indices = torch.argsort(scores, descending=True).tolist()
        chosen_action_idx = int(sorted_indices[0])
        result = self.controller.execute(
            question=question,
            gold_answer=gold,
            action_idx=chosen_action_idx,
            task_type=task_info["task_type"],
        )

        top_n = max(1, min(topk or self.service_config.topk, len(sorted_indices)))
        top_actions = []
        for idx in sorted_indices[:top_n]:
            action_desc = self.action_space.describe_action(int(idx))
            top_actions.append(
                {
                    "action_idx": int(idx),
                    "action": action_desc,
                    "label": describe_action(action_desc),
                    "score": float(scores[int(idx)].item()),
                }
            )

        record = {
            "request_id": str(uuid.uuid4()),
            "question": question,
            "request": question,
            "task": task_info,
            "gold": gold if gold else None,
            "policy": {
                "model_path": self.model_path,
                "api_mode": self.resolved_api_mode,
                "requested_api_mode": self.requested_api_mode,
                "embedding_model": self.service_config.embedding_model,
                "policy_source": (
                    "checkpoint" if self.model is not None and task_info["task_type"] == "math" else "heuristic"
                ),
                "policy_note": "checkpoint currently applies to math-like requests only",
            },
            "difficulty": difficulty,
            "chosen_action_idx": chosen_action_idx,
            "chosen_action": result.action_description,
            "chosen_action_label": describe_action(result.action_description),
            "top_actions": top_actions,
            "model_name": result.model_name,
            "pred": result.extracted_answer,
            "correct": (result.extracted_answer == gold) if gold else None,
            "final_text": result.final_text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "reward_breakdown": result.reward_breakdown if gold else None,
        }
        if log_request:
            self.request_store.append(record)
        return record

    def demo_batch(self, num_samples: int = 5) -> Dict[str, Any]:
        dataset = load_demo_dataset(n_samples=num_samples)
        records = [
            self.predict(
                question=sample["question"],
                gold=extract_gold_answer(sample["answer"]),
                log_request=True,
            )
            for sample in dataset
        ]
        return summarize_records(records, self.model_path, self.service_config)

    def metrics(self) -> Dict[str, Any]:
        return self.request_store.metrics()

    def recent_logs(self, limit: int = 50) -> Dict[str, Any]:
        rows = self.request_store.read_recent(limit=limit)
        return {
            "count": len(rows),
            "records": rows,
        }

    def add_feedback(self, request_record: Dict[str, Any], feedback: Dict[str, Any]) -> Dict[str, Any]:
        record = {
            "type": "feedback",
            "request_id": request_record.get("request_id"),
            "question": request_record.get("question", ""),
            "request": request_record.get("request") or request_record.get("question", ""),
            "task": request_record.get("task"),
            "pred": request_record.get("pred"),
            "gold": request_record.get("gold"),
            "chosen_action_idx": request_record.get("chosen_action_idx"),
            "feedback": feedback,
        }
        return self.request_store.append(record)


def summarize_records(
    records: List[Dict[str, Any]],
    model_path: str,
    service_config: ServiceConfig,
) -> Dict[str, Any]:
    total = len(records)
    correct = sum(1 for row in records if row["correct"])
    reward_sum = sum(row["reward_breakdown"]["total_reward"] for row in records if row["reward_breakdown"])
    token_sum = sum(row["total_tokens"] for row in records)
    action_hist: Dict[str, int] = {}

    for row in records:
        key = str(row["chosen_action_idx"])
        action_hist[key] = action_hist.get(key, 0) + 1

    return {
        "summary": {
            "model_path": model_path,
            "api_mode": resolve_api_mode(service_config.api_mode),
            "requested_api_mode": service_config.api_mode,
            "embedding_model": service_config.embedding_model,
            "num_samples": total,
            "accuracy": correct / max(total, 1),
            "avg_reward": reward_sum / max(total, 1),
            "avg_total_tokens": token_sum / max(total, 1),
            "chosen_action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        },
        "records": records,
    }


def make_args(**kwargs):
    return SimpleNamespace(**kwargs)


def resolve_api_mode(api_mode: str) -> str:
    if api_mode == "auto":
        return "openai" if os.getenv("OPENAI_API_KEY") else "mock"
    return api_mode
