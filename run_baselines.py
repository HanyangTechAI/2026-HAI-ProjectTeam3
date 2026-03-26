import os

import torch

from configs import TrainConfig
from src.baselines import (
    run_fixed_action_baseline,
    run_random_baseline,
    run_rl_policy_baseline,
)
from src.data import load_gsm8k_subset
from src.embedder import build_embedding_cache
from src.llm_client import build_llm_client
from src.policy import PromptActorCritic
from src.prompt_space import PromptSpace
from src.utils import ensure_dir, print_artifact_summary, save_csv, save_json


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

    results = []
    artifact_paths = {
        "output_dir": cfg.output_dir,
        "test_embedding_cache": os.path.join(cfg.output_dir, cfg.test_embedding_cache),
    }

    random_result = run_random_baseline(
        dataset=test_ds,
        prompt_space=prompt_space,
        llm_client=llm_client,
        cfg=cfg,
        seed=cfg.seed,
    )
    results.append(
        {
            "name": random_result["name"],
            "reward": random_result["reward"],
            "accuracy": random_result["accuracy"],
        }
    )
    random_output_path = os.path.join(cfg.output_dir, cfg.random_json)
    save_json(random_output_path, random_result)
    artifact_paths["random_baseline"] = random_output_path

    for action_idx in cfg.fixed_action_indices:
        fixed_result = run_fixed_action_baseline(
            dataset=test_ds,
            fixed_action_idx=action_idx,
            prompt_space=prompt_space,
            llm_client=llm_client,
            cfg=cfg,
        )
        results.append(
            {
                "name": fixed_result["name"],
                "reward": fixed_result["reward"],
                "accuracy": fixed_result["accuracy"],
                "action_idx": fixed_result["action_idx"],
            }
        )
        fixed_output_path = os.path.join(cfg.output_dir, f"fixed_{action_idx}.json")
        save_json(fixed_output_path, fixed_result)
        artifact_paths[f"fixed_action_{action_idx}"] = fixed_output_path

    model_path = os.path.join(cfg.output_dir, "prompt_policy.pt")
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model = PromptActorCritic(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            n_actions=checkpoint["n_actions"],
            dropout=checkpoint["dropout"],
        )
        model.load_state_dict(checkpoint["state_dict"])
        model = model.to(device)

        rl_result = run_rl_policy_baseline(
            dataset=test_ds,
            embeddings=test_embeddings,
            model=model,
            prompt_space=prompt_space,
            llm_client=llm_client,
            cfg=cfg,
            device=device,
        )
        results.append(
            {
                "name": rl_result["name"],
                "reward": rl_result["reward"],
                "accuracy": rl_result["accuracy"],
            }
        )
        rl_output_path = os.path.join(cfg.output_dir, cfg.rl_policy_json)
        save_json(rl_output_path, rl_result)
        artifact_paths["rl_policy_eval"] = rl_output_path

    leaderboard = sorted(results, key=lambda x: (x["accuracy"], x["reward"]), reverse=True)

    leaderboard_json_path = os.path.join(cfg.output_dir, cfg.leaderboard_json)
    leaderboard_csv_path = os.path.join(cfg.output_dir, cfg.leaderboard_csv)
    save_json(leaderboard_json_path, leaderboard)
    save_csv(leaderboard_csv_path, leaderboard)
    artifact_paths["leaderboard_json"] = leaderboard_json_path
    artifact_paths["leaderboard_csv"] = leaderboard_csv_path

    print("[RESULT] leaderboard")
    for row in leaderboard:
        print(row)
    print_artifact_summary("saved artifacts", artifact_paths)


if __name__ == "__main__":
    main()