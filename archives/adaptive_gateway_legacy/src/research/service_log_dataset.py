import json
from typing import Any, Dict, List

from src.controller.state_encoder import build_full_state


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def save_json(path: str, data: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_service_rows(rows: List[Dict[str, Any]]):
    request_rows = [row for row in rows if row.get("type") != "feedback"]
    feedback_rows = [row for row in rows if row.get("type") == "feedback"]
    feedback_by_request_id: Dict[str, List[Dict[str, Any]]] = {}

    for row in feedback_rows:
        request_id = row.get("request_id")
        if not request_id:
            continue
        feedback_by_request_id.setdefault(str(request_id), []).append(row)

    return request_rows, feedback_rows, feedback_by_request_id


def feedback_rating(feedback_rows: List[Dict[str, Any]]) -> str | None:
    if not feedback_rows:
        return None

    votes = []
    for row in feedback_rows:
        payload = row.get("feedback") or {}
        rating = str(payload.get("rating", "")).lower()
        if rating in {"up", "good", "positive", "1", "true"}:
            votes.append(1)
        elif rating in {"down", "bad", "negative", "-1", "false"}:
            votes.append(-1)

    if not votes:
        return None
    return "up" if sum(votes) >= 0 else "down"


def alternative_actions(row: Dict[str, Any]) -> List[int]:
    chosen = int(row["chosen_action_idx"])
    alternatives = []
    for item in row.get("top_actions", []):
        idx = int(item["action_idx"])
        if idx != chosen and idx not in alternatives:
            alternatives.append(idx)

    for idx in [0, 2, 4, 7]:
        if idx != chosen and idx not in alternatives:
            alternatives.append(idx)

    return alternatives


def request_text(row: Dict[str, Any]) -> str:
    return str(row.get("request") or row.get("question") or row.get("input") or "")


def build_pair(
    row: Dict[str, Any],
    preferred_action_idx: int,
    rejected_action_idx: int,
    pair_type: str,
    embedding_model: str,
) -> Dict[str, Any]:
    full_state = build_full_state(
        question=request_text(row),
        model_name=embedding_model,
        normalize_embedding=True,
    )
    return {
        "request": request_text(row),
        "question": request_text(row),
        "task": row.get("task"),
        "state_features": full_state["handcrafted"],
        "state_embedding": full_state["embedding"],
        "preferred_action_idx": int(preferred_action_idx),
        "rejected_action_idx": int(rejected_action_idx),
        "pair_type": pair_type,
        "source_request_id": row.get("request_id"),
        "chosen_action_idx": row.get("chosen_action_idx"),
        "correct": row.get("correct"),
        "reward_breakdown": row.get("reward_breakdown"),
    }


def build_feedback_preference_pairs(
    service_rows: List[Dict[str, Any]],
    embedding_model: str = "hashing:384",
    max_pairs_per_request: int = 3,
) -> Dict[str, Any]:
    request_rows, feedback_rows, feedback_by_request_id = split_service_rows(service_rows)
    pairs = []
    skipped = 0

    for row in request_rows:
        if not request_text(row) or "chosen_action_idx" not in row:
            skipped += 1
            continue

        chosen = int(row["chosen_action_idx"])
        alternatives = alternative_actions(row)[:max_pairs_per_request]
        if not alternatives:
            skipped += 1
            continue

        rating = feedback_rating(feedback_by_request_id.get(str(row.get("request_id")), []))
        correct = row.get("correct")

        if rating == "up" or (rating is None and correct is True):
            for alt in alternatives:
                pairs.append(
                    build_pair(
                        row=row,
                        preferred_action_idx=chosen,
                        rejected_action_idx=alt,
                        pair_type="service_positive",
                        embedding_model=embedding_model,
                    )
                )
        elif rating == "down" or (rating is None and correct is False):
            for alt in alternatives:
                pairs.append(
                    build_pair(
                        row=row,
                        preferred_action_idx=alt,
                        rejected_action_idx=chosen,
                        pair_type="service_negative",
                        embedding_model=embedding_model,
                    )
                )
        else:
            skipped += 1

    summary = {
        "num_service_rows": len(service_rows),
        "num_request_rows": len(request_rows),
        "num_feedback_rows": len(feedback_rows),
        "num_pairs": len(pairs),
        "num_skipped_requests": skipped,
        "embedding_model": embedding_model,
        "max_pairs_per_request": max_pairs_per_request,
    }
    return {
        "summary": summary,
        "pairs": pairs,
    }
