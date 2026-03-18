import torch

from configs import TrainConfig
from src.baselines import run_rl_policy_baseline
from src.data import load_gsm8k_subset
from src.llm_client import build_llm_client
from src.policy import PromptPolicy
from src.prompt_space import PromptSpace


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

    policy = PromptPolicy(input_dim=5, hidden_dim=32, n_actions=len(prompt_space))
    policy.load_state_dict(torch.load("prompt_policy.pt", map_location=device))
    policy = policy.to(device)

    result = run_rl_policy_baseline(
        dataset=test_ds,
        policy=policy,
        prompt_space=prompt_space,
        llm_client=llm_client,
        cfg=cfg,
        device=device,
    )

    print(result)


if __name__ == "__main__":
    main()