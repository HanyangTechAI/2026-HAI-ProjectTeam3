import argparse
import json
from collections import Counter

from src.research.service_log_dataset import (
    build_feedback_preference_pairs,
    load_jsonl,
    split_service_rows,
)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze service logs as research data for policy optimization."
    )
    parser.add_argument("--log_path", default="outputs/service_requests.jsonl")
    parser.add_argument("--embedding_model", default="hashing:384")
    args = parser.parse_args()

    rows = load_jsonl(args.log_path)
    request_rows, feedback_rows, feedback_by_request_id = split_service_rows(rows)
    pair_output = build_feedback_preference_pairs(
        service_rows=rows,
        embedding_model=args.embedding_model,
    )
    pairs = pair_output["pairs"]

    action_hist = Counter(str(row.get("chosen_action_idx")) for row in request_rows)
    route_hist = Counter(
        str((row.get("chosen_action") or {}).get("model_route", "unknown"))
        for row in request_rows
    )
    pair_type_hist = Counter(row.get("pair_type", "unknown") for row in pairs)
    preferred_hist = Counter(str(row.get("preferred_action_idx")) for row in pairs)
    rejected_hist = Counter(str(row.get("rejected_action_idx")) for row in pairs)

    labeled_rows = [row for row in request_rows if row.get("correct") is not None]
    correct = sum(1 for row in labeled_rows if row.get("correct"))

    summary = {
        "num_raw_rows": len(rows),
        "num_request_rows": len(request_rows),
        "num_feedback_rows": len(feedback_rows),
        "num_requests_with_feedback": len(feedback_by_request_id),
        "num_preference_pairs": len(pairs),
        "accuracy_on_labeled_requests": (
            correct / len(labeled_rows) if labeled_rows else None
        ),
        "chosen_action_hist": dict(action_hist.most_common()),
        "route_hist": dict(route_hist.most_common()),
        "pair_type_hist": dict(pair_type_hist.most_common()),
        "preferred_action_hist": dict(preferred_hist.most_common()),
        "rejected_action_hist": dict(rejected_hist.most_common()),
        "research_notes": [
            "More diverse service traffic is needed before trusting feedback-trained checkpoints.",
            "If preferred_action_hist collapses to one action, collect harder negatives or add cost-aware pairs.",
            "User feedback should be paired with gold/correctness when available to reduce noisy preference labels.",
        ],
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
