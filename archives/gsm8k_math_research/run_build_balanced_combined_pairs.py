import json
import os
import random

from configs import TrainConfig


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    cfg = TrainConfig()
    random.seed(getattr(cfg, "seed", 42))

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)
    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    efficiency_ratio = getattr(cfg, "balanced_efficiency_ratio", 0.5)

    hard_path = os.path.join(cfg.output_dir, f"hard_negative_preference_pairs_{suffix}.json")
    eff_path = os.path.join(cfg.output_dir, f"efficiency_preference_pairs_{suffix}.json")

    hard_pairs = load_json(hard_path)
    eff_pairs = load_json(eff_path)

    for row in hard_pairs:
        row["pair_type"] = "hard_negative"
    for row in eff_pairs:
        row["pair_type"] = "efficiency"

    max_eff = int(len(hard_pairs) * efficiency_ratio)
    if len(eff_pairs) > max_eff:
        eff_pairs = random.sample(eff_pairs, k=max_eff)

    combined = hard_pairs + eff_pairs
    random.shuffle(combined)

    summary = {
        "split": split,
        "start_idx": start_idx,
        "num_samples": num_samples,
        "num_hard_negative_pairs": len(hard_pairs),
        "num_efficiency_pairs_used": len(eff_pairs),
        "num_total_pairs": len(combined),
        "balanced_efficiency_ratio": efficiency_ratio,
    }

    save_json(
        os.path.join(cfg.output_dir, f"balanced_combined_pairs_{suffix}.json"),
        combined,
    )
    save_json(
        os.path.join(cfg.output_dir, f"balanced_combined_summary_{suffix}.json"),
        summary,
    )

    print(summary)


if __name__ == "__main__":
    main()