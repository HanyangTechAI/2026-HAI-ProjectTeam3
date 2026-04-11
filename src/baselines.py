import random
from dataclasses import asdict

import torch
from tqdm import tqdm

from src.data import extract_gold_answer
from src.evaluator import evaluate_single_action_on_sample
from src.reward import compute_reward, extract_pred_answer, is_correct


def run_random_baseline(dataset, prompt_space, llm_client, cfg, seed: int = 42):
    rng = random.Random(seed)

    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    action_hist = {}
    records = []

    for sample in tqdm(dataset, desc="random_baseline", leave=False):
        action_idx = rng.randrange(len(prompt_space))
        record = evaluate_single_action_on_sample(
            sample=sample,
            action_idx=action_idx,
            prompt_space=prompt_space,
            llm_client=llm_client,
            cfg=cfg,
        )
        total_reward += record.reward
        total_correct += int(record.correct)
        total_prompt_tokens += int(record.prompt_tokens)
        total_completion_tokens += int(record.completion_tokens)
        action_hist[action_idx] = action_hist.get(action_idx, 0) + 1
        records.append(asdict(record))

    n = len(dataset)
    return {
        "name": "random",
        "reward": total_reward / n,
        "accuracy": total_correct / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        "records": records,
    }


def run_fixed_action_baseline(dataset, fixed_action_idx: int, prompt_space, llm_client, cfg):
    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    records = []

    for sample in tqdm(dataset, desc=f"fixed_{fixed_action_idx}", leave=False):
        record = evaluate_single_action_on_sample(
            sample=sample,
            action_idx=fixed_action_idx,
            prompt_space=prompt_space,
            llm_client=llm_client,
            cfg=cfg,
        )
        total_prompt_tokens += int(record.prompt_tokens)
        total_completion_tokens += int(record.completion_tokens)
        total_reward += record.reward
        total_correct += int(record.correct)
        records.append(asdict(record))

    n = len(dataset)
    return {
        "name": f"fixed_{fixed_action_idx}",
        "action_idx": fixed_action_idx,
        "reward": total_reward / n,
        "accuracy": total_correct / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "records": records,
    }


@torch.no_grad()
def run_rl_policy_baseline(dataset, embeddings, model, prompt_space, llm_client, cfg, device: str = "cpu"):
    model.eval()
    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    action_hist = {}
    records = []

    for idx, sample in enumerate(tqdm(dataset, desc="rl_policy", leave=False)):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        x = embeddings[idx].to(device).unsqueeze(0)
        action, pred_value = model.greedy_action(x)
        action_idx = int(action.item())

        prompt = prompt_space.render_prompt(action_idx, question)
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
        action_hist[action_idx] = action_hist.get(action_idx, 0) + 1

        records.append(
            {
                "question": question,
                "gold": gold,
                "pred": pred,
                "correct": correct,
                "reward": reward,
                "predicted_value": float(pred_value.item()),
                "prompt_tokens": response.prompt_tokens,
                "completion_tokens": response.completion_tokens,
                "total_tokens": response.total_tokens,
                "action_idx": action_idx,
                "raw_text": response.text,
            }
        )

    n = len(dataset)
    return {
        "name": "rl_policy",
        "reward": total_reward / n,
        "accuracy": total_correct / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        "records": records,
    }


def run_exhaustive_search(dataset, prompt_space, llm_client, cfg):
    per_action = {
        idx: {"total_reward": 0.0, "total_correct": 0, "count": 0}
        for idx in range(len(prompt_space))
    }

    oracle_total_reward = 0.0
    oracle_total_correct = 0
    oracle_records = []

    for sample in tqdm(dataset, desc="exhaustive", leave=False):
        best_record = None

        for action_idx in range(len(prompt_space)):
            record = evaluate_single_action_on_sample(
                sample=sample,
                action_idx=action_idx,
                prompt_space=prompt_space,
                llm_client=llm_client,
                cfg=cfg,
            )

            per_action[action_idx]["total_reward"] += record.reward
            per_action[action_idx]["total_correct"] += int(record.correct)
            per_action[action_idx]["count"] += 1

            if best_record is None or record.reward > best_record.reward:
                best_record = record

        oracle_total_reward += best_record.reward
        oracle_total_correct += int(best_record.correct)
        oracle_records.append(asdict(best_record))

    global_action_scores = []
    for action_idx, stats in per_action.items():
        count = max(1, stats["count"])
        global_action_scores.append(
            {
                "action_idx": action_idx,
                "avg_reward": stats["total_reward"] / count,
                "avg_accuracy": stats["total_correct"] / count,
            }
        )

    global_action_scores.sort(key=lambda x: (x["avg_reward"], x["avg_accuracy"]), reverse=True)
    best_global_action = global_action_scores[0]

    n = len(dataset)
    return {
        "name": "exhaustive",
        "oracle_reward": oracle_total_reward / n,
        "oracle_accuracy": oracle_total_correct / n,
        "best_global_action": best_global_action,
        "all_action_scores": global_action_scores,
        "oracle_records": oracle_records,
    }