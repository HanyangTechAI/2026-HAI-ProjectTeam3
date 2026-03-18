import torch

from configs import TrainConfig
from src.data import load_gsm8k_subset
from src.llm_client import build_llm_client
from src.policy import PromptPolicy
from src.prompt_space import PromptSpace
from src.trainer import PromptRLTrainer
from src.utils import set_seed


def main():
    cfg = TrainConfig()
    set_seed(cfg.seed)

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

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = trainer.train_epoch(train_ds)
        eval_metrics = trainer.evaluate(test_ds)

        print(
            f"[EPOCH {epoch}] "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_reward={train_metrics['reward']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
            f"eval_reward={eval_metrics['reward']:.4f} "
            f"eval_acc={eval_metrics['accuracy']:.4f}"
        )
        print(f"[EPOCH {epoch}] top_actions={list(eval_metrics['action_hist'].items())[:5]}")

    torch.save(policy.state_dict(), "prompt_policy.pt")
    print("[INFO] saved policy -> prompt_policy.pt")


if __name__ == "__main__":
    main()