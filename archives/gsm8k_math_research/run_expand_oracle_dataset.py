import os
import random

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.data import load_gsm8k_subset, extract_gold_answer
from src.imitation.dataset_builder import save_json
from src.llm_client import build_llm_client


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(getattr(cfg, "seed", 42))

    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 50)

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=split,
        n_samples=start_idx + num_samples,
    )
    dataset = dataset.select(range(start_idx, start_idx + num_samples))

    llm_client = build_llm_client(cfg.api_mode)
    action_space = InferenceActionSpace()
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    all_records = []
    oracle_records = []

    total_correct = 0
    total_reward = 0.0

    for local_idx, sample in enumerate(dataset):
        global_sample_idx = start_idx + local_idx
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        per_action_results = []

        for action_idx in range(len(action_space)):
            result = controller.execute(
                question=question,
                gold_answer=gold,
                action_idx=action_idx,
            )

            pred = result.extracted_answer
            correct = pred == gold
            reward = result.reward_breakdown["total_reward"]

            row = {
                "global_sample_idx": global_sample_idx,
                "question": question,
                "gold": gold,
                "action_idx": action_idx,
                "action": result.action_description,
                "model_name": result.model_name,
                "raw_text": result.raw_text,
                "final_text": result.final_text,
                "pred": pred,
                "correct": correct,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "verification_used": result.verification_used,
                "format_ok": result.format_ok,
                "reward_breakdown": result.reward_breakdown,
            }

            all_records.append(row)
            per_action_results.append(row)

        best_row = max(
            per_action_results,
            key=lambda x: x["reward_breakdown"]["total_reward"],
        )

        oracle_records.append(best_row)
        total_correct += int(best_row["correct"])
        total_reward += best_row["reward_breakdown"]["total_reward"]

        print(
            f"[ORACLE] sample={global_sample_idx} "
            f"best_action={best_row['action_idx']} "
            f"correct={best_row['correct']} "
            f"reward={best_row['reward_breakdown']['total_reward']:.4f}"
        )

    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    summary = {
        "split": split,
        "start_idx": start_idx,
        "num_samples": num_samples,
        "oracle_accuracy": total_correct / max(num_samples, 1),
        "oracle_avg_reward": total_reward / max(num_samples, 1),
    }

    save_json(
        os.path.join(cfg.output_dir, f"expanded_all_records_{suffix}.json"),
        all_records,
    )
    save_json(
        os.path.join(cfg.output_dir, f"expanded_oracle_records_{suffix}.json"),
        oracle_records,
    )
    save_json(
        os.path.join(cfg.output_dir, f"expanded_oracle_summary_{suffix}.json"),
        summary,
    )

    print("[ORACLE SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()