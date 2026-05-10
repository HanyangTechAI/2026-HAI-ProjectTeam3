import os
import random

import numpy as np
import torch

from configs import TrainConfig
from archives.grpo.policy_model import GRPOPolicyNet
from archives.grpo.trainer import prepare_grpo_examples, run_grpo_epoch
from imitation.dataset_builder import load_json, save_json


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rollout_path = os.path.join(cfg.output_dir, "policy_improvement_rollout_records_explore.json")
    rollout_records = load_json(rollout_path)

    examples = prepare_grpo_examples(rollout_records)

    input_dim = 7 + len(rollout_records[0]["state_embedding"])
    num_actions = 12

    model = GRPOPolicyNet(
        input_dim=input_dim,
        hidden_dim=128,
        num_actions=num_actions,
        dropout=0.1,
    ).to(device)

    # imitation policy warm start 가능
    warm_start_path = os.path.join(cfg.output_dir, "imitation_policy_embedding.pt")
    if os.path.exists(warm_start_path):
        ckpt = torch.load(warm_start_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("[INFO] loaded warm start from imitation policy")
        except Exception as e:
            print(f"[WARN] warm start failed: {e}")

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    history = []
    epochs = 40

    for epoch in range(1, epochs + 1):
        out = run_grpo_epoch(
            model=model,
            examples=examples,
            optimizer=optimizer,
            device=device,
            entropy_coef=0.01,
        )

        row = {
            "epoch": epoch,
            "loss": out.loss,
            "mean_advantage": out.mean_advantage,
            "mean_reward": out.mean_reward,
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"loss={out.loss:.4f} "
            f"mean_adv={out.mean_advantage:.4f} "
            f"mean_reward={out.mean_reward:.4f}"
        )

    model_path = os.path.join(cfg.output_dir, "grpo_style_policy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "num_actions": num_actions,
        },
        model_path,
    )

    save_json(os.path.join(cfg.output_dir, "grpo_style_train_history.json"), history)

    print(f"[INFO] saved GRPO-style policy -> {model_path}")


if __name__ == "__main__":
    main()