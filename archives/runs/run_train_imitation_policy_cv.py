import math
import os
import random
from copy import deepcopy

import numpy as np
import torch

from configs import TrainConfig
from imitation.dataset_builder import load_json, save_json
from archives.policy_model import ImitationPolicyNet
from archives.trainer import (
    build_dataloader,
    compute_class_weights_from_examples,
    predict_actions,
    run_epoch,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_kfold_splits(examples, k: int):
    n = len(examples)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = math.ceil(n / k)
    folds = []

    for i in range(k):
        val_idx = indices[i * fold_size : (i + 1) * fold_size]
        train_idx = [idx for idx in indices if idx not in val_idx]
        if len(val_idx) == 0:
            continue
        folds.append((train_idx, val_idx))

    return folds


def subset_by_indices(examples, indices):
    return [examples[i] for i in indices]


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

    k = 5 if len(examples) >= 10 else len(examples)
    folds = make_kfold_splits(examples, k=k)

    num_actions = 12
    hidden_dim = 128
    dropout = 0.1
    lr = 1e-3
    epochs = 60
    batch_size = 8
    patience = 10

    cv_history = []
    fold_summaries = []
    all_val_predictions = []

    for fold_id, (train_idx, val_idx) in enumerate(folds, start=1):
        train_examples = subset_by_indices(examples, train_idx)
        val_examples = subset_by_indices(examples, val_idx)

        train_loader = build_dataloader(train_examples, batch_size=batch_size, shuffle=True)
        val_loader = build_dataloader(val_examples, batch_size=batch_size, shuffle=False)

        model = ImitationPolicyNet(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_actions=num_actions,
            dropout=dropout,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        class_weights = compute_class_weights_from_examples(
            train_examples,
            num_actions=num_actions,
        ).to(device)

        best_val_acc = -1.0
        best_val_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, epochs + 1):
            train_out = run_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                class_weights=class_weights,
            )
            val_out = run_epoch(
                model,
                val_loader,
                optimizer=None,
                device=device,
                class_weights=class_weights,
            )

            row = {
                "fold": fold_id,
                "epoch": epoch,
                "train_loss": train_out.loss,
                "train_accuracy": train_out.accuracy,
                "val_loss": val_out.loss,
                "val_accuracy": val_out.accuracy,
            }
            cv_history.append(row)

            print(
                f"[FOLD {fold_id}][EPOCH {epoch}] "
                f"train_loss={train_out.loss:.4f} "
                f"train_acc={train_out.accuracy:.4f} "
                f"val_loss={val_out.loss:.4f} "
                f"val_acc={val_out.accuracy:.4f}"
            )

            improved = False
            if val_out.accuracy > best_val_acc:
                improved = True
            elif val_out.accuracy == best_val_acc and val_out.loss < best_val_loss:
                improved = True

            if improved:
                best_val_acc = val_out.accuracy
                best_val_loss = val_out.loss
                best_state = deepcopy(model.state_dict())
                wait = 0
            else:
                wait += 1

            if wait >= patience:
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_preds = predict_actions(model, val_examples, device=device)

        match_count = 0
        for ex, pred in zip(val_examples, val_preds):
            match = int(ex["label_action_idx"] == pred)
            match_count += match
            all_val_predictions.append(
                {
                    "fold": fold_id,
                    "question": ex["question"],
                    "label_action_idx": ex["label_action_idx"],
                    "pred_action_idx": pred,
                    "match": match,
                }
            )

        final_val_acc = match_count / max(len(val_examples), 1)

        fold_summaries.append(
            {
                "fold": fold_id,
                "num_train": len(train_examples),
                "num_val": len(val_examples),
                "best_val_acc": best_val_acc,
                "best_val_loss": best_val_loss,
                "final_val_acc_from_predictions": final_val_acc,
            }
        )

    mean_best_val_acc = sum(x["best_val_acc"] for x in fold_summaries) / len(fold_summaries)
    mean_final_val_acc = (
        sum(x["final_val_acc_from_predictions"] for x in fold_summaries) / len(fold_summaries)
    )

    summary = {
        "num_examples": len(examples),
        "input_dim": input_dim,
        "embedding_dim": embedding_dim,
        "num_folds": len(fold_summaries),
        "mean_best_val_acc": mean_best_val_acc,
        "mean_final_val_acc": mean_final_val_acc,
        "fold_summaries": fold_summaries,
    }

    save_json(os.path.join(cfg.output_dir, "imitation_cv_history.json"), cv_history)
    save_json(os.path.join(cfg.output_dir, "imitation_cv_val_predictions.json"), all_val_predictions)
    save_json(os.path.join(cfg.output_dir, "imitation_cv_summary.json"), summary)

    print("\n[CV SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()