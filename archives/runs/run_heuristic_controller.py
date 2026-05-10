import json
import os

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from archives.heuristic_policy import HeuristicInferencePolicy
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import summarize_query_state
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
    heuristic_policy = HeuristicInferencePolicy(action_space=action_space)

    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    action_hist = {}
    records = []

    for sample_idx, sample in enumerate(dataset):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        chosen_action_idx = heuristic_policy.choose_action(question)
        result = controller.execute(
            question=question,
            gold_answer=gold,
            action_idx=chosen_action_idx,
        )

        correct = (result.extracted_answer == gold)
        reward = result.reward_breakdown["total_reward"]

        total_reward += reward
        total_correct += int(correct)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_tokens += result.total_tokens
        action_hist[chosen_action_idx] = action_hist.get(chosen_action_idx, 0) + 1

        record = {
            "sample_idx": sample_idx,
            "question": question,
            "gold": gold,
            "chosen_action_idx": chosen_action_idx,
            "chosen_action": result.action_description,
            "state_features": summarize_query_state(question),
            "model_name": result.model_name,
            "raw_text": result.raw_text,
            "final_text": result.final_text,
            "pred": result.extracted_answer,
            "correct": correct,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "verification_used": result.verification_used,
            "format_ok": result.format_ok,
            "reward_breakdown": result.reward_breakdown,
        }
        records.append(record)

        print(
            f"[SAMPLE {sample_idx}] "
            f"action={chosen_action_idx} "
            f"correct={correct} "
            f"reward={reward:.4f}"
        )

    n = len(dataset)
    summary = {
        "num_samples": n,
        "accuracy": total_correct / n,
        "avg_reward": total_reward / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "avg_total_tokens": total_tokens / n,
        "action_hist": {
            str(k): v for k, v in sorted(action_hist.items(), key=lambda x: x[1], reverse=True)
        },
    }

    save_json(os.path.join(cfg.output_dir, "heuristic_controller_records.json"), records)
    save_json(os.path.join(cfg.output_dir, "heuristic_controller_summary.json"), summary)

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()