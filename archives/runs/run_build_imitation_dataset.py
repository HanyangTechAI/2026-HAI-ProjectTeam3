import os

from configs import TrainConfig
from src.controller.state_encoder import build_full_state
from imitation.dataset_builder import load_json, save_json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    oracle_path = os.path.join(cfg.output_dir, "inference_policy_oracle_records.json")
    oracle_records = load_json(oracle_path)

    examples = []
    for row in oracle_records:
        question = row["question"]
        action_idx = row["action_idx"]

        full_state = build_full_state(
            question=question,
            model_name=cfg.embedding_model_name,
            normalize_embedding=True,
        )

        examples.append(
            {
                "question": question,
                "state_features": full_state["handcrafted"],
                "state_embedding": full_state["embedding"],
                "label_action_idx": action_idx,
                "gold": row.get("gold"),
                "oracle_reward": row.get("reward_breakdown", {}).get("total_reward"),
                "oracle_action": row.get("action"),
            }
        )

    out_path = os.path.join(cfg.output_dir, "imitation_dataset.json")
    save_json(out_path, examples)

    print(f"[INFO] built imitation dataset -> {out_path}")
    print(f"[INFO] num_examples = {len(examples)}")
    print(f"[INFO] embedding_dim = {len(examples[0]['state_embedding']) if examples else 0}")


if __name__ == "__main__":
    main()