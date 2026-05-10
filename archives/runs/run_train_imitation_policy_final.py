import os
import random

import numpy as np
import torch

from configs import TrainConfig
from imitation.dataset_builder import load_json, save_json
from archives.policy_model import ImitationPolicyNet
from archives.trainer import (
    build_dataloader,
    compute_class_weights_from_examples,
    run_epoch,
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

    imitation_path = os.path.join(cfg.output_dir, "imitation_dataset.json")
    examples = load_json(imitation_path)

    handcrafted_dim = 7
    embedding_dim = len(examples[0]["state_embedding"])
    input_dim = handcrafted_dim + embedding_dim
    num_actions = 12

    hidden_dim = 128
    dropout = 0.1
    lr = 1e-3
    epochs = 60
    batch_size = 8

    loader = build_dataloader(examples, batch_size=batch_size, shuffle=True)

    model = ImitationPolicyNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    class_weights = compute_class_weights_from_examples(
        examples,
        num_actions=num_actions,
    ).to(device)

    history = []

    for epoch in range(1, epochs + 1):
        train_out = run_epoch(
            model,
            loader,
            optimizer=optimizer,
            device=device,
            class_weights=class_weights,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_out.loss,
            "train_accuracy": train_out.accuracy,
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_out.loss:.4f} "
            f"train_acc={train_out.accuracy:.4f}"
        )

    model_path = os.path.join(cfg.output_dir, "imitation_policy_embedding.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_actions": num_actions,
            "embedding_dim": embedding_dim,
        },
        model_path,
    )

    save_json(os.path.join(cfg.output_dir, "imitation_final_train_history.json"), history)

    print(f"[INFO] saved final imitation model -> {model_path}")


if __name__ == "__main__":
    main()