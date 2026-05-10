import os
import random

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from imitation.dataset_builder import load_json, save_json
from src.preference.preference_model import ActionPreferenceNet
from src.preference.trainer import (
    build_preference_dataloader,
    run_preference_epoch,
)


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

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 100)
    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    data_path = os.path.join(cfg.output_dir, f"filtered_preference_pairs_{suffix}.json")
    examples = load_json(data_path)

    action_space = InferenceActionSpace()

    state_dim = 7 + len(examples[0]["state_embedding"])
    action_dim = 6

    model = ActionPreferenceNet(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    loader = build_preference_dataloader(
        examples=examples,
        action_space=action_space,
        batch_size=32,
        shuffle=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    epochs = 30
    history = []

    for epoch in range(1, epochs + 1):
        out = run_preference_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
        )

        row = {
            "epoch": epoch,
            "train_loss": out.loss,
            "pair_accuracy": out.pair_accuracy,
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={out.loss:.4f} "
            f"pair_acc={out.pair_accuracy:.4f}"
        )

    model_path = os.path.join(cfg.output_dir, f"action_preference_model_{suffix}.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "data_suffix": suffix,
        },
        model_path,
    )

    save_json(os.path.join(cfg.output_dir, f"action_preference_train_history_{suffix}.json"), history)

    print(f"[INFO] saved expanded preference model -> {model_path}")


if __name__ == "__main__":
    main()