import json
import os
import random

import numpy as np
import torch

from configs import TrainConfig
from imitation.dataset_builder import load_json, save_json
from archives.policy_model import ImitationPolicyNet
from archives.trainer import build_dataloader, predict_actions, run_epoch


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_examples(examples, train_ratio=0.8):
    n = len(examples)
    split_idx = max(1, int(n * train_ratio))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:] if split_idx < n else examples[-1:]
    return train_examples, val_examples


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    imitation_path = os.path.join(cfg.output_dir, "imitation_dataset.json")
    examples = load_json(imitation_path)

    random.shuffle(examples)
    train_examples, val_examples = split_examples(examples, train_ratio=0.8)

    batch_size = 8
    hidden_dim = 64
    lr = 1e-3
    epochs = 50
    input_dim = 7
    num_actions = 12

    train_loader = build_dataloader(train_examples, batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(val_examples, batch_size=batch_size, shuffle=False)

    model = ImitationPolicyNet(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_actions=num_actions,
        dropout=0.1,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = []
    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_out = run_epoch(model, train_loader, optimizer=optimizer, device=device)
        val_out = run_epoch(model, val_loader, optimizer=None, device=device)

        row = {
            "epoch": epoch,
            "train_loss": train_out.loss,
            "train_accuracy": train_out.accuracy,
            "val_loss": val_out.loss,
            "val_accuracy": val_out.accuracy,
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_out.loss:.4f} "
            f"train_acc={train_out.accuracy:.4f} "
            f"val_loss={val_out.loss:.4f} "
            f"val_acc={val_out.accuracy:.4f}"
        )

        if val_out.accuracy > best_val_acc:
            best_val_acc = val_out.accuracy
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    pred_train = predict_actions(model, train_examples, device=device)
    pred_val = predict_actions(model, val_examples, device=device)

    train_predictions = []
    for ex, pred in zip(train_examples, pred_train):
        train_predictions.append(
            {
                "question": ex["question"],
                "label_action_idx": ex["label_action_idx"],
                "pred_action_idx": pred,
                "match": int(ex["label_action_idx"] == pred),
            }
        )

    val_predictions = []
    for ex, pred in zip(val_examples, pred_val):
        val_predictions.append(
            {
                "question": ex["question"],
                "label_action_idx": ex["label_action_idx"],
                "pred_action_idx": pred,
                "match": int(ex["label_action_idx"] == pred),
            }
        )

    model_path = os.path.join(cfg.output_dir, "imitation_policy.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "num_actions": num_actions,
            "best_val_acc": best_val_acc,
        },
        model_path,
    )

    save_json(os.path.join(cfg.output_dir, "imitation_train_history.json"), history)
    save_json(os.path.join(cfg.output_dir, "imitation_train_predictions.json"), train_predictions)
    save_json(os.path.join(cfg.output_dir, "imitation_val_predictions.json"), val_predictions)

    print(f"[INFO] saved model -> {model_path}")
    print(f"[INFO] best_val_acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    main()