import argparse
import json
import os
import random

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import extract_gold_answer, load_gsm8k_subset
from src.demo_samples import load_demo_dataset
from src.llm_client import build_llm_client
from src.policy_utils import (
    compute_action_scores,
    compute_heuristic_action_scores,
    estimate_difficulty,
    load_preference_model,
    resolve_policy_checkpoint,
)
from src.reporting import render_demo_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def describe_action(action_desc: dict) -> str:
    verify = "verify" if action_desc["verify"] else "no-verify"
    return f"{action_desc['reasoning_budget']} / {action_desc['model_route']} / {verify}"


def load_eval_dataset(args, cfg: TrainConfig):
    if args.dataset == "demo":
        return load_demo_dataset(n_samples=args.num_samples, start_idx=args.start_idx)

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=args.split,
        n_samples=args.start_idx + args.num_samples,
    )
    return dataset.select(range(args.start_idx, args.start_idx + args.num_samples))


def run_one(question: str, gold: str, model, controller, action_space, cfg, args, device: str):
    full_state = build_full_state(
        question=question,
        model_name=args.embedding_model,
        normalize_embedding=True,
    )
    state_features = full_state["handcrafted"]
    state_embedding = full_state["embedding"]
    difficulty = estimate_difficulty(state_features)

    with torch.no_grad():
        if model is None:
            scores = compute_heuristic_action_scores(
                state_features=state_features,
                action_space=action_space,
                device=device,
            )
        else:
            scores = compute_action_scores(
                model=model,
                state_features=state_features,
                state_embedding=state_embedding,
                action_space=action_space,
                device=device,
            )

    sorted_indices = torch.argsort(scores, descending=True).tolist()
    chosen_action_idx = int(sorted_indices[0])
    result = controller.execute(
        question=question,
        gold_answer=gold,
        action_idx=chosen_action_idx,
    )

    top_actions = []
    for idx in sorted_indices[: args.topk]:
        action_desc = action_space.describe_action(int(idx))
        top_actions.append(
            {
                "action_idx": int(idx),
                "action": action_desc,
                "label": describe_action(action_desc),
                "score": float(scores[int(idx)].item()),
            }
        )

    return {
        "question": question,
        "gold": gold if gold else None,
        "difficulty": difficulty,
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": result.action_description,
        "chosen_action_label": describe_action(result.action_description),
        "top_actions": top_actions,
        "model_name": result.model_name,
        "pred": result.extracted_answer,
        "correct": (result.extracted_answer == gold) if gold else None,
        "final_text": result.final_text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "reward_breakdown": result.reward_breakdown if gold else None,
    }


def run_fixed_action(question: str, gold: str, action_idx: int, controller, action_space):
    result = controller.execute(
        question=question,
        gold_answer=gold,
        action_idx=action_idx,
    )
    return {
        "action_idx": action_idx,
        "action_label": describe_action(action_space.describe_action(action_idx)),
        "pred": result.extracted_answer,
        "correct": result.extracted_answer == gold,
        "total_tokens": result.total_tokens,
        "reward": result.reward_breakdown["total_reward"],
    }


def summarize_rollouts(name: str, action_label: str, rows: list[dict]) -> dict:
    total = len(rows)
    return {
        "name": name,
        "action_label": action_label,
        "num_samples": total,
        "accuracy": sum(1 for row in rows if row["correct"]) / max(total, 1),
        "avg_reward": sum(row["reward"] for row in rows) / max(total, 1),
        "avg_total_tokens": sum(row["total_tokens"] for row in rows) / max(total, 1),
    }


def evaluate_fixed_baselines(dataset, controller, action_space) -> list[dict]:
    baseline_actions = [0, 2, 4, 7]
    summaries = []

    for action_idx in baseline_actions:
        rows = []
        for sample in dataset:
            rows.append(
                run_fixed_action(
                    question=sample["question"],
                    gold=extract_gold_answer(sample["answer"]),
                    action_idx=action_idx,
                    controller=controller,
                    action_space=action_space,
                )
            )

        action_label = describe_action(action_space.describe_action(action_idx))
        summaries.append(
            summarize_rollouts(
                name=f"fixed_action_{action_idx}",
                action_label=action_label,
                rows=rows,
            )
        )

    return summaries


def print_single(record: dict, model_path: str):
    print("\n[RL INFERENCE POLICY DEMO]")
    print(f"checkpoint: {model_path}")
    print(f"question: {record['question']}")
    print(f"difficulty: {record['difficulty']['difficulty_level']} ({record['difficulty']['difficulty_score']:.3f})")
    print(f"chosen action: {record['chosen_action_idx']} | {record['chosen_action_label']}")
    print("top actions:")
    for row in record["top_actions"]:
        print(f"  - {row['action_idx']}: {row['label']} | score={row['score']:.4f}")
    print(f"prediction: {record['pred']}")
    if record["gold"] is not None:
        print(f"gold: {record['gold']} | correct={record['correct']}")
        print(f"reward: {record['reward_breakdown']['total_reward']:.4f}")
    print(f"tokens: prompt={record['prompt_tokens']}, completion={record['completion_tokens']}, total={record['total_tokens']}")


def main():
    parser = argparse.ArgumentParser(description="Demo for RL-based LLM inference policy optimization.")
    parser.add_argument("--mode", choices=["single", "batch"], default="batch")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--gold", type=str, default="")
    parser.add_argument("--dataset", choices=["demo", "gsm8k"], default="demo")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--api_mode", choices=["mock", "openai"], default="mock")
    parser.add_argument("--embedding_model", type=str, default="hashing:384")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument(
        "--policy_source",
        choices=["auto", "checkpoint", "heuristic"],
        default="auto",
    )
    parser.add_argument("--force_heuristic", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--save_html", type=str, default="")
    parser.add_argument("--no_baselines", action="store_true")
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy_source = args.policy_source
    if args.force_heuristic:
        policy_source = "heuristic"
    if policy_source == "auto" and args.api_mode == "mock" and args.dataset == "demo" and not args.checkpoint:
        policy_source = "heuristic"

    if policy_source == "heuristic":
        model_path = "heuristic_fallback_policy"
        model = None
    else:
        try:
            model_path = resolve_policy_checkpoint(cfg.output_dir, args.checkpoint or None)
            model, _ = load_preference_model(model_path, device=device)
        except FileNotFoundError:
            if args.checkpoint or policy_source == "checkpoint":
                raise
            model_path = "heuristic_fallback_policy"
            model = None

    action_space = InferenceActionSpace()
    llm_client = build_llm_client(args.api_mode)
    controller = AdaptiveInferenceController(llm_client=llm_client, action_space=action_space)

    if args.mode == "single":
        if args.question:
            question = args.question.strip()
            gold = args.gold.strip()
        else:
            sample = load_demo_dataset(n_samples=1)[0]
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

        record = run_one(question, gold, model, controller, action_space, cfg, args, device)
        print_single(record, model_path)
        output = {"model_path": model_path, "record": record}
    else:
        dataset = load_eval_dataset(args, cfg)
        records = []
        for sample in dataset:
            records.append(
                run_one(
                    question=sample["question"],
                    gold=extract_gold_answer(sample["answer"]),
                    model=model,
                    controller=controller,
                    action_space=action_space,
                    cfg=cfg,
                    args=args,
                    device=device,
                )
            )

        total = len(records)
        correct = sum(1 for row in records if row["correct"])
        reward_sum = sum(row["reward_breakdown"]["total_reward"] for row in records)
        token_sum = sum(row["total_tokens"] for row in records)
        action_hist = {}
        for row in records:
            key = str(row["chosen_action_idx"])
            action_hist[key] = action_hist.get(key, 0) + 1

        summary = {
            "model_path": model_path,
            "api_mode": args.api_mode,
            "embedding_model": args.embedding_model,
            "dataset": args.dataset,
            "num_samples": total,
            "accuracy": correct / max(total, 1),
            "avg_reward": reward_sum / max(total, 1),
            "avg_total_tokens": token_sum / max(total, 1),
            "chosen_action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }
        baseline_summaries = []
        if not args.no_baselines:
            policy_rows = [
                {
                    "correct": bool(row["correct"]),
                    "reward": row["reward_breakdown"]["total_reward"],
                    "total_tokens": row["total_tokens"],
                }
                for row in records
            ]
            baseline_summaries.append(
                summarize_rollouts(
                    name="checkpoint_policy" if model is not None else "heuristic_dynamic_policy",
                    action_label="dynamic action selection",
                    rows=policy_rows,
                )
            )
            baseline_summaries.extend(evaluate_fixed_baselines(dataset, controller, action_space))

        print("\n[RL INFERENCE POLICY BATCH DEMO]")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if baseline_summaries:
            print("\n[BASELINE COMPARISON]")
            print(json.dumps(baseline_summaries, ensure_ascii=False, indent=2))
        output = {"summary": summary, "baselines": baseline_summaries, "records": records}

    if args.save_json:
        parent = os.path.dirname(args.save_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\nsaved: {args.save_json}")

    if args.save_html:
        parent = os.path.dirname(args.save_html)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_html, "w", encoding="utf-8") as f:
            f.write(render_demo_report(output))
        print(f"saved: {args.save_html}")


if __name__ == "__main__":
    main()
