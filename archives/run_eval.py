import os

import torch

from archives.configs import TrainConfig
from data import load_gsm8k_subset
from archives.embedder import build_embedding_cache
from archives.llm_client import build_llm_client
from archives.policy import PromptActorCritic
from archives.prompt_space import PromptSpace
from archives.trainer import PromptPPOTrainer
from archives.utils import ensure_dir, print_artifact_summary, save_json


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_device = "cuda" if torch.cuda.is_available() else "cpu"

    test_ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.test_samples,
    )
    
    test_questions = [sample["question"] for sample in test_ds]

    test_embeddings = build_embedding_cache(
        questions=test_questions,
        cache_path=os.path.join(cfg.output_dir, cfg.test_embedding_cache),
        model_name=cfg.embedding_model_name,
        device=embedding_device,
        normalize_embeddings=cfg.normalize_embeddings,
        batch_size=cfg.embedding_batch_size,
        force_rebuild=False,
    )
    
    checkpoint_path = os.path.join(cfg.output_dir, "best_model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    prompt_space = PromptSpace()
    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    model = PromptActorCritic(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_instructions=checkpoint["num_instructions"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    
    trainer = PromptPPOTrainer(
        model=model,
        llm_client=llm_client,
        prompt_space=prompt_space,
        train_config=cfg,
        device=device,
    )
    
    eval_metrics = trainer.evaluate(test_ds, test_embeddings)

    result = {
        "model_type": checkpoint.get("model_type", "unknown"),
        "best_epoch": checkpoint.get("best_epoch"),
        "best_eval_acc_during_training": checkpoint.get("best_eval_acc"),
        "best_eval_reward_during_training": checkpoint.get("best_eval_reward"),
        "test_metrics": eval_metrics,
    }

    output_path = os.path.join(cfg.output_dir, cfg.rl_policy_json)
    save_json(output_path, result)

    print("[RESULT] PPO eval")
    print(result)
    print(f"[INFO] saved -> {output_path}")
    print_artifact_summary(
        "saved artifacts",
        {
            "output_dir": cfg.output_dir,
            "test_embedding_cache": os.path.join(cfg.output_dir, cfg.test_embedding_cache),
            "rl_policy_eval": output_path,
        },
    )

if __name__ == "__main__":
    main()