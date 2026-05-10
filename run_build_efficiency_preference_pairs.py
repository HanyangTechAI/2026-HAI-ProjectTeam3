import json
import os
from collections import defaultdict

from configs import TrainConfig
from src.controller.state_encoder import build_full_state


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def action_cost_tier(action_desc: dict) -> str:
    reasoning = action_desc["reasoning_budget"]
    route = action_desc["model_route"]
    verify = action_desc["verify"]

    if route == "small" and reasoning == "none" and not verify:
        return "low"

    if route == "small" and reasoning == "short" and not verify:
        return "low"

    if route == "small" and verify:
        return "mid"

    if route == "large" and reasoning == "short" and not verify:
        return "mid"

    if route == "large" and verify:
        return "high"

    if route == "large" and reasoning == "long":
        return "high"

    if reasoning == "long":
        return "mid"

    return "mid"


def cost_rank(tier: str) -> int:
    if tier == "low":
        return 0
    if tier == "mid":
        return 1
    return 2


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)

    reward_margin = getattr(cfg, "efficiency_reward_margin", 0.01)
    min_cost_gap = getattr(cfg, "efficiency_min_cost_gap", 1)

    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    all_records_path = os.path.join(
        cfg.output_dir,
        f"expanded_all_records_{suffix}.json",
    )
    all_records = load_json(all_records_path)

    grouped = defaultdict(list)
    for row in all_records:
        grouped[row["global_sample_idx"]].append(row)

    efficiency_pairs = []

    total_questions = 0
    usable_questions = 0
    total_candidate_pairs = 0
    kept_pairs = 0

    for global_idx, rows in grouped.items():
        total_questions += 1

        question = rows[0]["question"]
        gold = rows[0]["gold"]

        full_state = build_full_state(
            question=question,
            model_name=cfg.embedding_model_name,
            normalize_embedding=True,
        )
        state_features = full_state["handcrafted"]
        state_embedding = full_state["embedding"]

        correct_rows = [r for r in rows if r["correct"]]
        if len(correct_rows) < 2:
            continue

        usable_questions += 1

        for i in range(len(correct_rows)):
            for j in range(len(correct_rows)):
                if i == j:
                    continue

                a = correct_rows[i]
                b = correct_rows[j]

                a_tier = action_cost_tier(a["action"])
                b_tier = action_cost_tier(b["action"])

                a_rank = cost_rank(a_tier)
                b_rank = cost_rank(b_tier)

                if a_rank >= b_rank:
                    continue

                total_candidate_pairs += 1

                reward_diff = (
                    a["reward_breakdown"]["total_reward"]
                    - b["reward_breakdown"]["total_reward"]
                )
                cost_gap = b_rank - a_rank

                if reward_diff < reward_margin:
                    continue
                if cost_gap < min_cost_gap:
                    continue

                efficiency_pairs.append(
                    {
                        "global_sample_idx": global_idx,
                        "question": question,
                        "gold": gold,
                        "state_features": state_features,
                        "state_embedding": state_embedding,
                        "preferred_action_idx": a["action_idx"],
                        "rejected_action_idx": b["action_idx"],
                        "preferred_reward": a["reward_breakdown"]["total_reward"],
                        "rejected_reward": b["reward_breakdown"]["total_reward"],
                        "reward_diff": reward_diff,
                        "preferred_correct": True,
                        "rejected_correct": True,
                        "preferred_action": a["action"],
                        "rejected_action": b["action"],
                        "preferred_cost_tier": a_tier,
                        "rejected_cost_tier": b_tier,
                        "pair_type": "efficiency",
                    }
                )
                kept_pairs += 1

    summary = {
        "split": split,
        "start_idx": start_idx,
        "num_samples": num_samples,
        "total_questions": total_questions,
        "usable_questions": usable_questions,
        "total_candidate_pairs": total_candidate_pairs,
        "num_pairs": kept_pairs,
        "efficiency_reward_margin": reward_margin,
        "efficiency_min_cost_gap": min_cost_gap,
    }

    save_json(
        os.path.join(cfg.output_dir, f"efficiency_preference_pairs_{suffix}.json"),
        efficiency_pairs,
    )
    save_json(
        os.path.join(cfg.output_dir, f"efficiency_preference_summary_{suffix}.json"),
        summary,
    )

    print(summary)


if __name__ == "__main__":
    main()