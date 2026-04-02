import os

import torch

from configs import TrainConfig
from src.baselines import run_rl_policy_baseline
from src.data import load_gsm8k_subset
from src.embedder import build_embedding_cache
from src.llm_client import build_llm_client
from src.policy import PromptPolicy
from src.prompt_space import PromptSpace
from src.utils import ensure_dir, print_artifact_summary, save_json


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

    prompt_space = PromptSpace()
    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    checkpoint = torch.load(os.path.join(cfg.output_dir, "prompt_policy.pt"), map_location=device)
    policy = PromptPolicy(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        n_actions=checkpoint["n_actions"],
        dropout=checkpoint["dropout"],
    )
    policy.load_state_dict(checkpoint["state_dict"])
    policy = policy.to(device)

    result = run_rl_policy_baseline(
        dataset=test_ds,
        embeddings=test_embeddings,
        policy=policy,
        prompt_space=prompt_space,
        llm_client=llm_client,
        cfg=cfg,
        device=device,
    )

    output_path = os.path.join(cfg.output_dir, cfg.rl_policy_json)
    save_json(output_path, result)

    print(result)
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
