import json
import os

from configs import TrainConfig


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def maybe_load(path: str):
    if os.path.exists(path):
        return load_json(path)
    return None


def main():
    cfg = TrainConfig()

    candidates = {
        "best_fixed_reference": None,  # 수동 기준용
        "imitation_controller": os.path.join(cfg.output_dir, "imitation_controller_summary(2).json"),
        "improved_imitation_v2": os.path.join(cfg.output_dir, "improved_imitation_controller_summary_v2(1).json"),
        "preference_controller_small_eval": os.path.join(cfg.output_dir, "preference_controller_summary(1).json"),
    }

    # holdout summary 자동 탐색
    holdout_split = getattr(cfg, "holdout_split", "test")
    holdout_start_idx = getattr(cfg, "holdout_start_idx", 0)
    holdout_num_samples = getattr(cfg, "holdout_num_samples", 50)
    holdout_suffix = f"{holdout_split}_{holdout_start_idx}_{holdout_start_idx + holdout_num_samples - 1}"
    holdout_path = os.path.join(cfg.output_dir, f"preference_holdout_summary_{holdout_suffix}.json")

    if os.path.exists(holdout_path):
        candidates["preference_controller_holdout"] = holdout_path

    rows = []

    # 수동 reference
    rows.append(
        {
            "name": "best_fixed_reference",
            "accuracy": 0.5333333333333333,
            "avg_reward": 0.4419,
            "source": "manual_reference",
        }
    )

    for name, path in candidates.items():
        if path is None:
            continue
        data = maybe_load(path)
        if data is None:
            continue

        rows.append(
            {
                "name": name,
                "accuracy": data.get("accuracy"),
                "avg_reward": data.get("avg_reward"),
                "avg_prompt_tokens": data.get("avg_prompt_tokens"),
                "avg_completion_tokens": data.get("avg_completion_tokens"),
                "avg_total_tokens": data.get("avg_total_tokens"),
                "chosen_action_hist": data.get("chosen_action_hist"),
                "source": path,
            }
        )

    rows.sort(
        key=lambda x: (
            -1 if x["accuracy"] is None else -x["accuracy"],
            -1 if x["avg_reward"] is None else -x["avg_reward"],
        )
    )

    out_path = os.path.join(cfg.output_dir, "policy_summary_comparison.json")
    save_json(out_path, rows)

    print("[COMPARISON]")
    for row in rows:
        print(row)

    print(f"[INFO] saved -> {out_path}")


if __name__ == "__main__":
    main()