import json
import os
from collections import defaultdict

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.state_encoder import build_full_state


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def action_cost_tier(action):
    r = action["reasoning_budget"]
    m = action["model_route"]
    v = action["verify"]

    if m == "small" and r == "none" and not v:
        return "low"

    if m == "small" and r == "short":
        return "low"

    if m == "small" and v:
        return "mid"

    if m == "large" and r == "short" and not v:
        return "mid"

    if r == "long" and m == "small":
        return "mid"

    if m == "large":
        return "high"

    return "high"


def required_margin_for_preferred(
    preferred_action: dict,
    base_margin: float,
    extra_mid_margin: float,
    extra_high_margin: float,
) -> float:
    tier = action_cost_tier(preferred_action)

    if tier == "high":
        return base_margin + extra_high_margin
    if tier == "mid":
        return base_margin + extra_mid_margin
    return base_margin


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)

    base_margin = getattr(cfg, "cost_aware_base_margin", 0.0)
    extra_mid_margin = getattr(cfg, "cost_aware_mid_margin", 0.02)
    extra_high_margin = getattr(cfg, "cost_aware_high_margin", 0.05)

    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    all_records_path = os.path.join(
        cfg.output_dir,
        f"expanded_all_records_{suffix}.json",
    )
    all_records = load_json(all_records_path)

    grouped = defaultdict(list)
    for row in all_records:
        grouped[row["global_sample_idx"]].append(row)

    cost_aware_pairs = []
    best_action_dataset = []

    total_questions = 0
    usable_questions = 0
    kept_pairs = 0
    skipped_by_margin = 0

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
        wrong_rows = [r for r in rows if not r["correct"]]

        if len(correct_rows) == 0 or len(wrong_rows) == 0:
            continue

        usable_questions += 1

        correct_rows = sorted(
            correct_rows,
            key=lambda x: x["reward_breakdown"]["total_reward"],
            reverse=True,
        )
        wrong_rows = sorted(
            wrong_rows,
            key=lambda x: x["reward_breakdown"]["total_reward"],
            reverse=True,
        )

        best_correct = correct_rows[0]
        best_action_dataset.append(
            {
                "global_sample_idx": global_idx,
                "question": question,
                "gold": gold,
                "state_features": state_features,
                "state_embedding": state_embedding,
                "label_action_idx": best_correct["action_idx"],
                "oracle_reward": best_correct["reward_breakdown"]["total_reward"],
                "oracle_action": best_correct["action"],
            }
        )

        for better in correct_rows:
            required_margin = required_margin_for_preferred(
                preferred_action=better["action"],
                base_margin=base_margin,
                extra_mid_margin=extra_mid_margin,
                extra_high_margin=extra_high_margin,
            )

            for worse in wrong_rows:
                reward_diff = (
                    better["reward_breakdown"]["total_reward"]
                    - worse["reward_breakdown"]["total_reward"]
                )

                if reward_diff < required_margin:
                    skipped_by_margin += 1
                    continue

                cost_aware_pairs.append(
                    {
                        "global_sample_idx": global_idx,
                        "question": question,
                        "gold": gold,
                        "state_features": state_features,
                        "state_embedding": state_embedding,
                        "preferred_action_idx": better["action_idx"],
                        "rejected_action_idx": worse["action_idx"],
                        "preferred_reward": better["reward_breakdown"]["total_reward"],
                        "rejected_reward": worse["reward_breakdown"]["total_reward"],
                        "reward_diff": reward_diff,
                        "preferred_correct": True,
                        "rejected_correct": False,
                        "preferred_action": better["action"],
                        "rejected_action": worse["action"],
                        "preferred_cost_tier": action_cost_tier(better["action"]),
                        "required_margin": required_margin,
                    }
                )
                kept_pairs += 1

    save_json(
        os.path.join(cfg.output_dir, f"cost_aware_hard_negative_pairs_{suffix}.json"),
        cost_aware_pairs,
    )
    save_json(
        os.path.join(cfg.output_dir, f"cost_aware_best_action_dataset_{suffix}.json"),
        best_action_dataset,
    )

    summary = {
        "split": split,
        "start_idx": start_idx,
        "num_samples": num_samples,
        "total_questions": total_questions,
        "usable_questions": usable_questions,
        "num_pairs": kept_pairs,
        "skipped_by_margin": skipped_by_margin,
        "base_margin": base_margin,
        "extra_mid_margin": extra_mid_margin,
        "extra_high_margin": extra_high_margin,
    }

    save_json(
        os.path.join(cfg.output_dir, f"cost_aware_hard_negative_summary_{suffix}.json"),
        summary,
    )

    print(summary)


if __name__ == "__main__":
    main()