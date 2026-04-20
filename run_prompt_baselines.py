import json
import os
from dataclasses import asdict

from configs import TrainConfig
from src.data import load_gsm8k_subset, extract_gold_answer
from src.llm_client import build_llm_client
from src.prompt_space import PromptSpace, PromptAction
from src.reward import compute_reward, extract_pred_answer, is_correct


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def evaluate_fixed_prompt(dataset, prompt_space, instruction_idx, llm_client, cfg):
    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    records = []

    action = PromptAction(instruction_idx=instruction_idx)

    for sample in dataset:
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        prompt = prompt_space.render_prompt(action, question)
        response = llm_client.generate(prompt)

        pred = extract_pred_answer(response.text)
        correct = is_correct(pred, gold)

        reward = compute_reward(
            pred=pred,
            gold=gold,
            completion_tokens=response.completion_tokens,
            reward_correct=cfg.reward_correct,
            reward_wrong=cfg.reward_wrong,
            completion_token_penalty_coef=cfg.completion_token_penalty_coef,
        )

        total_reward += reward
        total_correct += int(correct)
        total_prompt_tokens += int(response.prompt_tokens)
        total_completion_tokens += int(response.completion_tokens)

        records.append(
            {
                "question": question,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "reward": reward,
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "raw_text": response.text,
            }
        )

    n = len(dataset)
    return {
        "instruction_idx": instruction_idx,
        "few_shot_mode": prompt_space.few_shot_mode,
        "reward": total_reward / n,
        "accuracy": total_correct / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "instruction": prompt_space.describe_action(action)["instruction"],
        "records": records,
    }


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.test_samples,
    )

    llm_client = build_llm_client(
        api_mode=cfg.api_mode,
        model_name=cfg.model_name,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
    )

    few_shot_modes = [
        "no_shot",
        "one_shot",
        "two_shot",
        "gsm8k_style_two_shot",
    ]

    all_results = []

    for few_shot_mode in few_shot_modes:
        prompt_space = PromptSpace(few_shot_mode=few_shot_mode)

        for instruction_idx in range(prompt_space.num_instructions):
            result = evaluate_fixed_prompt(
                dataset=dataset,
                prompt_space=prompt_space,
                instruction_idx=instruction_idx,
                llm_client=llm_client,
                cfg=cfg,
            )
            all_results.append(result)

            filename = f"baseline_inst_{instruction_idx}_{few_shot_mode}.json"
            save_json(os.path.join(cfg.output_dir, filename), result)

            print(
                f"[BASELINE] mode={few_shot_mode} inst={instruction_idx} "
                f"acc={result['accuracy']:.4f} reward={result['reward']:.4f} "
                f"prompt_tok={result['avg_prompt_tokens']:.2f} "
                f"completion_tok={result['avg_completion_tokens']:.2f}"
            )

    leaderboard = sorted(
        [
            {
                "instruction_idx": r["instruction_idx"],
                "few_shot_mode": r["few_shot_mode"],
                "accuracy": r["accuracy"],
                "reward": r["reward"],
                "avg_prompt_tokens": r["avg_prompt_tokens"],
                "avg_completion_tokens": r["avg_completion_tokens"],
                "instruction": r["instruction"],
            }
            for r in all_results
        ],
        key=lambda x: (x["accuracy"], x["reward"]),
        reverse=True,
    )

    save_json(os.path.join(cfg.output_dir, "prompt_baseline_leaderboard.json"), leaderboard)

    print("\n[TOP 10]")
    for row in leaderboard[:10]:
        print(row)


if __name__ == "__main__":
    main()