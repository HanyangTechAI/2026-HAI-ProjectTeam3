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


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    split = getattr(cfg, "oracle_split", cfg.train_split if hasattr(cfg, "train_split") else "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)

    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"
    all_records_path = os.path.join(cfg.output_dir, f"expanded_all_records_{suffix}.json")
    all_records = load_json(all_records_path)

    grouped = defaultdict(list)
    for row in all_records:
        grouped[row["global_sample_idx"]].append(row)

    preference_pairs = []
    best_action_dataset = []

    for global_idx, rows in grouped.items():
        rows = sorted(rows, key=lambda x: x["reward_breakdown"]["total_reward"], reverse=True)

        question = rows[0]["question"]
        gold = rows[0]["gold"]

        full_state = build_full_state(
            question=question,
            model_name=cfg.embedding_model_name,
            normalize_embedding=True,
        )
        state_features = full_state["handcrafted"]
        state_embedding = full_state["embedding"]

        best = rows[0]
        best_action_dataset.append(
            {
                "global_sample_idx": global_idx,
                "question": question,
                "gold": gold,
                "state_features": state_features,
                "state_embedding": state_embedding,
                "label_action_idx": best["action_idx"],
                "oracle_reward": best["reward_breakdown"]["total_reward"],
                "oracle_action": best["action"],
            }
        )

        for worse in rows[1:]:
            preference_pairs.append(
                {
                    "global_sample_idx": global_idx,
                    "question": question,
                    "gold": gold,
                    "state_features": state_features,
                    "state_embedding": state_embedding,
                    "preferred_action_idx": best["action_idx"],
                    "rejected_action_idx": worse["action_idx"],
                    "preferred_reward": best["reward_breakdown"]["total_reward"],
                    "rejected_reward": worse["reward_breakdown"]["total_reward"],
                }
            )

    save_json(os.path.join(cfg.output_dir, f"expanded_preference_pairs_{suffix}.json"), preference_pairs)
    save_json(os.path.join(cfg.output_dir, f"expanded_best_action_dataset_{suffix}.json"), best_action_dataset)

    print(f"[INFO] best-action samples: {len(best_action_dataset)}")
    print(f"[INFO] preference pairs: {len(preference_pairs)}")


if __name__ == "__main__":
    main()