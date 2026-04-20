import os
import random
import numpy as np
import torch

from configs import TrainConfig
from src.data import load_gsm8k_subset
from src.embedder import build_embedding_cache
from src.llm_client import build_llm_client
from src.policy import PromptActorCritic
from src.prompt_space import PromptSpace
from src.trainer import PromptPPOTrainer
from src.utils import ensure_dir, print_artifact_summary, save_csv, save_json, set_seed

def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"

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

    train_questions = [sample["question"] for sample in train_ds]
    test_questions = [sample["question"] for sample in test_ds]

    train_embedding_path = os.path.join(cfg.output_dir, cfg.train_embedding_cache)
    test_embedding_path = os.path.join(cfg.output_dir, cfg.test_embedding_cache)

    train_embeddings = build_embedding_cache(
        questions=train_questions,
        cache_path=train_embedding_path,
        model_name=cfg.embedding_model_name,
        device=embedding_device,
        normalize_embeddings=cfg.normalize_embeddings,
        batch_size=cfg.embedding_batch_size,
        force_rebuild=False,
    )
    test_embeddings = build_embedding_cache(
        questions=test_questions,
        cache_path=test_embedding_path,
        model_name=cfg.embedding_model_name,
        device=embedding_device,
        normalize_embeddings=cfg.normalize_embeddings,
        batch_size=cfg.embedding_batch_size,
        force_rebuild=False,
    )

    input_dim = int(train_embeddings.shape[1])

    prompt_space = PromptSpace()
    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    model = PromptActorCritic(
        input_dim=input_dim,
        hidden_dim=cfg.policy_hidden_dim,
        num_instructions=prompt_space.num_instructions,
        dropout=cfg.policy_dropout,
    )

    trainer = PromptPPOTrainer(
        model=model,
        prompt_space=prompt_space,
        llm_client=llm_client,
        train_config=cfg,
        device=device,
    )

    history = []
    best_eval_acc = -1.0
    best_eval_reward = -1e9
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = trainer.train_epoch(train_ds, train_embeddings)
        eval_metrics = trainer.evaluate(test_ds, test_embeddings)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_policy_loss": train_metrics["policy_loss"],
            "train_value_loss": train_metrics["value_loss"],
            "train_entropy": train_metrics["entropy"],
            "train_reward": train_metrics["reward"],
            "train_accuracy": train_metrics["accuracy"],
            "train_avg_prompt_tokens": train_metrics["avg_prompt_tokens"],
            "train_avg_completion_tokens": train_metrics["avg_completion_tokens"],
            "eval_reward": eval_metrics["reward"],
            "eval_accuracy": eval_metrics["accuracy"],
            "eval_avg_prompt_tokens": eval_metrics["avg_prompt_tokens"],
            "eval_avg_completion_tokens": eval_metrics["avg_completion_tokens"],
            "train_top_action": next(iter(train_metrics["action_hist"].keys()), None),
            "eval_top_action": next(iter(eval_metrics["action_hist"].keys()), None),
        }
        history.append(row)

        print(
            f"[EPOCH {epoch}] "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f} "
            f"eval_reward={eval_metrics['reward']:.4f} "
            f"eval_completion_tok={eval_metrics['avg_completion_tokens']:.2f}"
        )
        print(f"[EPOCH {epoch}] eval_top_actions={list(eval_metrics['action_hist'].items())[:5]}")

        is_better = False
        if eval_metrics["accuracy"] > best_eval_acc:
            is_better = True
        elif eval_metrics["accuracy"] == best_eval_acc and eval_metrics["reward"] > best_eval_reward:
            is_better = True

        if is_better:
            best_eval_acc = eval_metrics["accuracy"]
            best_eval_reward = eval_metrics["reward"]
            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": trainer.model.state_dict(),
                    "input_dim": input_dim,
                    "hidden_dim": cfg.policy_hidden_dim,
                    "dropout": cfg.policy_dropout,
                    "num_instructions": prompt_space.num_instructions,
                    "model_type": "ppo_actor_critic",
                    "best_epoch": best_epoch,
                    "best_eval_acc": best_eval_acc,
                    "best_eval_reward": best_eval_reward,
                },
                os.path.join(cfg.output_dir, "best_model.pt"),
            )

    save_json(os.path.join(cfg.output_dir, cfg.train_log_json), history)
    save_csv(os.path.join(cfg.output_dir, cfg.train_log_csv), history)

    print(f"[INFO] best_epoch={best_epoch}")
    print(f"[INFO] best_eval_acc={best_eval_acc:.4f}")
    print(f"[INFO] best_eval_reward={best_eval_reward:.4f}")

    model_path = os.path.join(cfg.output_dir, "newest_model.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": cfg.policy_hidden_dim,
            "dropout": cfg.policy_dropout,
            "num_instructions": prompt_space.num_instructions,
            "model_type": "ppo_actor_critic",
        },
        model_path,
    )
    print(f"[INFO] saved model -> {model_path}")

    train_history_json_path = os.path.join(cfg.output_dir, cfg.train_log_json)
    train_history_csv_path = os.path.join(cfg.output_dir, cfg.train_log_csv)
    save_json(train_history_json_path, history)
    save_csv(train_history_csv_path, history)

    print_artifact_summary(
        "saved artifacts",
        {
            "output_dir": cfg.output_dir,
            "train_embedding_cache": train_embedding_path,
            "test_embedding_cache": test_embedding_path,
            "policy_checkpoint": model_path,
            "train_history_json": train_history_json_path,
            "train_history_csv": train_history_csv_path,
        },
    )


if __name__ == "__main__":
    main()