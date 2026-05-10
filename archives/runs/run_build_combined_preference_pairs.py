import json
import os

from configs import TrainConfig


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    cfg = TrainConfig()

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)
    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    hard_path = os.path.join(cfg.output_dir, f"hard_negative_preference_pairs_{suffix}.json")
    eff_path = os.path.join(cfg.output_dir, f"efficiency_preference_pairs_{suffix}.json")

    hard_pairs = load_json(hard_path)
    eff_pairs = load_json(eff_path)

    # hard-negative에 pair_type 추가
    for row in hard_pairs:
        row["pair_type"] = "hard_negative"

    combined = hard_pairs + eff_pairs

    summary = {
        "split": split,
        "start_idx": start_idx,
        "num_samples": num_samples,
        "num_hard_negative_pairs": len(hard_pairs),
        "num_efficiency_pairs": len(eff_pairs),
        "num_total_pairs": len(combined),
    }

    save_json(
        os.path.join(cfg.output_dir, f"combined_preference_pairs_{suffix}.json"),
        combined,
    )
    save_json(
        os.path.join(cfg.output_dir, f"combined_preference_summary_{suffix}.json"),
        summary,
    )

    print(summary)


if __name__ == "__main__":
    main()