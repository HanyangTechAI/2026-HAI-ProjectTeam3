import os
import torch

from configs import TrainConfig
from src.baselines import (
    run_fixed_action_baseline,
    run_random_baseline,
    run_rl_policy_baseline,
)
from src.data import load_gsm8k_subset
from src.llm_client import build_llm_client
from src.policy import PromptPolicy
from src.prompt_space import PromptSpace
from src.utils import ensure_dir, save_json


def main():
    cfg = TrainConfig()
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    ensure_dir("outputs")

    results = []

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
            "action_hist": random_result["action_hist"],
        }
    )
    save_json("outputs/random_baseline.json", random_result)

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
                "action_desc": prompt_space.describe_action(action_idx),
            }
        )
        save_json(f"outputs/fixed_{action_idx}.json", fixed_result)

    if os.path.exists("prompt_policy.pt"):
        policy = PromptPolicy(input_dim=5, hidden_dim=32, n_actions=len(prompt_space))
        policy.load_state_dict(torch.load("prompt_policy.pt", map_location=device))
        policy = policy.to(device)

        rl_result = run_rl_policy_baseline(
            dataset=test_ds,
            policy=policy,
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
                "action_hist": rl_result["action_hist"],
            }
        )
        save_json("outputs/rl_policy.json", rl_result)

    leaderboard = sorted(results, key=lambda x: (x["accuracy"], x["reward"]), reverse=True)
    save_json("outputs/leaderboard.json", leaderboard)

    print("[RESULT] leaderboard")
    for row in leaderboard:
        print(row)


if __name__ == "__main__":
    main()