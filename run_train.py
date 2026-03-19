import os

import torch

from configs import TrainConfig
from src.data import load_gsm8k_subset
from src.llm_client import build_llm_client
from src.policy import PromptPolicy
from src.prompt_space import PromptSpace
from src.trainer import PromptRLTrainer
from src.utils import ensure_dir, save_csv, save_json, set_seed


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.train_split,
        n_samples=cfg.train_samples,
    )

    test_ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.test_samples,
    )

    prompt_space = PromptSpace()
    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    policy = PromptPolicy(
        input_dim=5,
        hidden_dim=32,
        n_actions=len(prompt_space),
    )

    trainer = PromptRLTrainer(
        policy=policy,
        prompt_space=prompt_space,
        llm_client=llm_client,
        train_config=cfg,
        device=device,
    )

    print(f"[INFO] device={device}")
    print(f"[INFO] train_samples={len(train_ds)} test_samples={len(test_ds)}")
    print(f"[INFO] action_space={len(prompt_space)}")

    history = []

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = trainer.train_epoch(train_ds)
        eval_metrics = trainer.evaluate(test_ds)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_reward": train_metrics["reward"],
            "train_accuracy": train_metrics["accuracy"],
            "train_avg_tokens": train_metrics["avg_tokens"],
            "eval_reward": eval_metrics["reward"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_avg_tokens": eval_metrics["avg_tokens"],
            "train_top_action": next(iter(train_metrics["action_hist"].keys()), None),
            "eval_top_action": next(iter(eval_metrics["action_hist"].keys()), None),
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_reward={train_metrics['reward']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_reward={eval_metrics['reward']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )
        print(f"[EPOCH {epoch}] train_top_actions={list(train_metrics['action_hist'].items())[:5]}")
        print(f"[EPOCH {epoch}] eval_top_actions={list(eval_metrics['action_hist'].items())[:5]}")

    model_path = os.path.join(cfg.output_dir, "prompt_policy.pt")
    torch.save(policy.state_dict(), model_path)
    print(f"[INFO] saved policy -> {model_path}")

    train_history_json_path = os.path.join(cfg.output_dir, cfg.train_log_json)
    train_history_csv_path = os.path.join(cfg.output_dir, cfg.train_log_csv)
    save_json(train_history_json_path, history)
    save_csv(train_history_csv_path, history)

    print(f"[INFO] saved train history -> {train_history_json_path}")
    print(f"[INFO] saved train history -> {train_history_csv_path}")


if __name__ == "__main__":
    main()