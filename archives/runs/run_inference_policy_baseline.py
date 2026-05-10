import json
import os
from collections import defaultdict

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.data import load_gsm8k_subset, extract_gold_answer
from src.llm_client import build_llm_client


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.test_samples,
    )

    llm_client = build_llm_client(cfg.api_mode)
    action_space = InferenceActionSpace()
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    action_stats = {
        idx: {
            "action_idx": idx,
            "action": action_space.describe_action(idx),
            "total_reward": 0.0,
            "total_correct": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "total_tokens": 0,
            "count": 0,
        }
        for idx in range(len(action_space))
    }

    per_sample_best = []
    all_records = []

    for sample_idx, sample in enumerate(dataset):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        best_result = None

        for action_idx in range(len(action_space)):
            result = controller.execute(
                question=question,
                gold_answer=gold,
                action_idx=action_idx,
            )

            pred = result.extracted_answer
            correct = pred == gold

            record = {
                "sample_idx": sample_idx,
                "question": question,
                "gold": gold,
                "action_idx": action_idx,
                "action": result.action_description,
                "model_name": result.model_name,
                "raw_text": result.raw_text,
                "final_text": result.final_text,
                "pred": pred,
                "correct": correct,
                "total_tokens": result.total_tokens,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "verification_used": result.verification_used,
                "format_ok": result.format_ok,
                "reward_breakdown": result.reward_breakdown,
            }
            all_records.append(record)

            action_stats[action_idx]["total_reward"] += result.reward_breakdown["total_reward"]
            action_stats[action_idx]["total_correct"] += int(correct)
            action_stats[action_idx]["total_prompt_tokens"] += result.prompt_tokens
            action_stats[action_idx]["total_completion_tokens"] += result.completion_tokens
            action_stats[action_idx]["total_tokens"] += result.total_tokens
            action_stats[action_idx]["count"] += 1

            if best_result is None or result.reward_breakdown["total_reward"] > best_result["reward_breakdown"]["total_reward"]:
                best_result = record

        per_sample_best.append(best_result)

        print(
            f"[SAMPLE {sample_idx}] "
            f"best_action={best_result['action_idx']} "
            f"correct={best_result['correct']} "
            f"reward={best_result['reward_breakdown']['total_reward']:.4f}"
        )

    leaderboard = []
    for idx, stats in action_stats.items():
        n = max(1, stats["count"])
        leaderboard.append(
            {
                "action_idx": idx,
                "action": stats["action"],
                "avg_reward": stats["total_reward"] / n,
                "accuracy": stats["total_correct"] / n,
                "avg_prompt_tokens": stats["total_prompt_tokens"] / n,
                "avg_completion_tokens": stats["total_completion_tokens"] / n,
                "avg_total_tokens": stats["total_tokens"] / n,
            }
        )

    leaderboard.sort(key=lambda x: (x["accuracy"], x["avg_reward"]), reverse=True)

    oracle_accuracy = sum(int(x["correct"]) for x in per_sample_best) / len(per_sample_best)
    oracle_reward = sum(x["reward_breakdown"]["total_reward"] for x in per_sample_best) / len(per_sample_best)

    summary = {
        "num_actions": len(action_space),
        "num_samples": len(dataset),
        "oracle_accuracy": oracle_accuracy,
        "oracle_reward": oracle_reward,
        "leaderboard": leaderboard,
    }

    save_json(os.path.join(cfg.output_dir, "inference_policy_action_leaderboard.json"), leaderboard)
    save_json(os.path.join(cfg.output_dir, "inference_policy_oracle_records.json"), per_sample_best)
    save_json(os.path.join(cfg.output_dir, "inference_policy_all_records.json"), all_records)
    save_json(os.path.join(cfg.output_dir, "inference_policy_summary.json"), summary)

    print("\n[TOP ACTIONS]")
    for row in leaderboard[:10]:
        print(row)

    print("\n[SUMMARY]")
    print(
        {
            "oracle_accuracy": oracle_accuracy,
            "oracle_reward": oracle_reward,
        }
    )


if __name__ == "__main__":
    main()