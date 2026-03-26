import os

from configs import TrainConfig
from src.baselines import run_exhaustive_search
from src.data import load_gsm8k_subset
from src.llm_client import build_llm_client
from src.prompt_space import PromptSpace
from src.utils import ensure_dir, print_artifact_summary, save_json


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.exhaustive_samples,
    )

    prompt_space = PromptSpace()
    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    result = run_exhaustive_search(
        dataset=ds,
        prompt_space=prompt_space,
        llm_client=llm_client,
        cfg=cfg,
    )

    best_global_action_idx = result["best_global_action"]["action_idx"]
    result["best_global_action_description"] = prompt_space.describe_action(best_global_action_idx)

    output_path = os.path.join(cfg.output_dir, cfg.exhaustive_json)
    save_json(output_path, result)

    print("[RESULT] exhaustive search summary")
    print(
        {
            "oracle_reward": result["oracle_reward"],
            "oracle_accuracy": result["oracle_accuracy"],
            "best_global_action": result["best_global_action"],
            "best_global_action_description": result["best_global_action_description"],
        }
    )
    print_artifact_summary(
        "saved artifacts",
        {
            "output_dir": cfg.output_dir,
            "exhaustive_result": output_path,
        },
    )


if __name__ == "__main__":
    main()