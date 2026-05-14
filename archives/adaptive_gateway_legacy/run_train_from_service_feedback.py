import argparse
import os
import random

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.imitation.dataset_builder import load_json, save_json
from src.preference.preference_model import ActionPreferenceNet
from src.preference.trainer import build_preference_dataloader, run_preference_epoch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(
        description="Train a preference model from service feedback pairs."
    )
    parser.add_argument("--pairs_path", default="outputs/service_feedback_pairs.json")
    parser.add_argument("--output_model", default="outputs/action_preference_model_service_feedback.pt")
    parser.add_argument("--history_path", default="outputs/service_feedback_train_history.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)

    data = load_json(args.pairs_path)
    examples = data["pairs"] if isinstance(data, dict) and "pairs" in data else data
    if not examples:
        raise ValueError(f"No service feedback pairs found in {args.pairs_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
        batch_size=args.batch_size,
        shuffle=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history = []

    for epoch in range(1, args.epochs + 1):
        result = run_preference_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
        )
        row = {
            "epoch": epoch,
            "train_loss": result.loss,
            "pair_accuracy": result.pair_accuracy,
        }
        history.append(row)
        print(
            f"[EPOCH {epoch}] "
            f"loss={result.loss:.4f} "
            f"pair_acc={result.pair_accuracy:.4f}"
        )

    parent = os.path.dirname(args.output_model)
    if parent:
        os.makedirs(parent, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "state_dim": state_dim,
            "action_dim": action_dim,
            "data_source": args.pairs_path,
            "num_pairs": len(examples),
            "model_type": "service_feedback_preference",
        },
        args.output_model,
    )
    save_json(args.history_path, history)
    print(f"[INFO] saved model -> {args.output_model}")
    print(f"[INFO] saved history -> {args.history_path}")


if __name__ == "__main__":
    main()
