import argparse
import os

from src.research.service_log_dataset import (
    build_feedback_preference_pairs,
    load_jsonl,
    save_json,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build preference pairs from service request and feedback logs."
    )
    parser.add_argument("--log_path", default="outputs/service_requests.jsonl")
    parser.add_argument("--output", default="outputs/service_feedback_pairs.json")
    parser.add_argument("--embedding_model", default="hashing:384")
    parser.add_argument("--max_pairs_per_request", type=int, default=3)
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        raise FileNotFoundError(f"Service log not found: {args.log_path}")

    rows = load_jsonl(args.log_path)
    output = build_feedback_preference_pairs(
        service_rows=rows,
        embedding_model=args.embedding_model,
        max_pairs_per_request=args.max_pairs_per_request,
    )

    parent = os.path.dirname(args.output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    save_json(args.output, output)

    print("[SERVICE FEEDBACK PAIRS]")
    print(output["summary"])
    print(f"[INFO] saved -> {args.output}")


if __name__ == "__main__":
    main()
