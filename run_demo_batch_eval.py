import argparse
import json
import os
import random
from typing import Dict, List

import numpy as np
import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import load_gsm8k_subset, extract_gold_answer
from src.llm_client import build_llm_client
from src.preference.action_encoder import encode_action_features
from src.preference.preference_model import ActionPreferenceNet


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_state_vector(state_features: dict, state_embedding: list) -> List[float]:
    handcrafted = [
        state_features["normalized_length"],
        state_features["normalized_word_count"],
        state_features["normalized_digit_count"],
        state_features["has_percent"],
        state_features["has_money"],
        state_features["has_ratio_words"],
        state_features["has_multistep_hint"],
    ]
    return handcrafted + state_embedding


def build_state_tensor(state_features: dict, state_embedding: list, device: str) -> torch.Tensor:
    vec = build_state_vector(state_features, state_embedding)
    return torch.tensor([vec], dtype=torch.float32, device=device)


def compute_action_scores(
    model: ActionPreferenceNet,
    state_features: dict,
    state_embedding: list,
    action_space: InferenceActionSpace,
    device: str,
) -> torch.Tensor:
    state_x = build_state_tensor(
        state_features=state_features,
        state_embedding=state_embedding,
        device=device,
    )

    scores = []
    for action_idx in range(len(action_space)):
        action_vec = encode_action_features(action_idx, action_space)
        action_x = torch.tensor([action_vec], dtype=torch.float32, device=device)
        score = model(state_x, action_x).squeeze()
        scores.append(score)

    return torch.stack(scores, dim=0)


def resolve_model_path(cfg: TrainConfig) -> str:
    train_split = getattr(cfg, "oracle_split", "train")
    train_start_idx = getattr(cfg, "oracle_start_idx", 0)
    train_num_samples = getattr(cfg, "oracle_num_samples", 50)
    train_suffix = f"{train_split}_{train_start_idx}_{train_start_idx + train_num_samples - 1}"

    candidate_paths = [
        os.path.join(cfg.output_dir, f"action_preference_model_cost_aware_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_balanced_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_hard_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, "action_preference_model.pt"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    import glob

    candidate_patterns = [
        os.path.join(cfg.output_dir, f"action_preference_model_cost_aware_{train_suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_balanced_{train_suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_hard_{train_suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_{train_suffix}*.pt"),
        os.path.join(cfg.output_dir, "action_preference_model*.pt"),
    ]

    matches = []
    for pattern in candidate_patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(
            f"No preference model checkpoint found in '{cfg.output_dir}'. "
            f"Tried exact paths: {candidate_paths} "
            f"and patterns: {candidate_patterns}"
        )

    priority_order = [
        "action_preference_model_cost_aware_",
        "action_preference_model_balanced_",
        "action_preference_model_hard_",
        "action_preference_model_",
    ]

    def priority_key(path: str):
        name = os.path.basename(path)
        for i, prefix in enumerate(priority_order):
            if name.startswith(prefix):
                return (i, len(name))
        return (999, len(name))

    matches.sort(key=priority_key)
    chosen = matches[0]
    print(f"[INFO] resolved model path: {chosen}")
    return chosen


def estimate_difficulty(state_features: Dict[str, float]) -> Dict[str, float]:
    score = 0.0
    score += 0.20 * state_features["normalized_length"]
    score += 0.20 * state_features["normalized_word_count"]
    score += 0.15 * state_features["normalized_digit_count"]
    score += 0.15 * state_features["has_ratio_words"]
    score += 0.20 * state_features["has_multistep_hint"]
    score += 0.05 * state_features["has_percent"]
    score += 0.05 * state_features["has_money"]

    score = max(0.0, min(score, 1.0))

    if score < 0.33:
        level = "easy"
    elif score < 0.66:
        level = "medium"
    else:
        level = "hard"

    return {
        "difficulty_score": score,
        "difficulty_level": level,
    }


def bucket_key(action_desc: Dict) -> str:
    return (
        f"{action_desc['reasoning_budget']}"
        f" | {action_desc['model_route']}"
        f" | verify={action_desc['verify']}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_json", type=str, default="")
    parser.add_argument("--save_records", type=str, default="")
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = resolve_model_path(cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    action_space = InferenceActionSpace()
    llm_client = build_llm_client(cfg.api_mode)
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=args.split,
        n_samples=args.start_idx + args.num_samples,
    )
    dataset = dataset.select(range(args.start_idx, args.start_idx + args.num_samples))

    total_correct = 0
    total_reward = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0

    action_hist = {}
    difficulty_hist = {"easy": 0, "medium": 0, "hard": 0}
    action_by_difficulty = {
        "easy": {},
        "medium": {},
        "hard": {},
    }

    records = []

    with torch.no_grad():
        for i, sample in enumerate(dataset):
            global_idx = args.start_idx + i
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            full_state = build_full_state(
                question=question,
                model_name=cfg.embedding_model_name,
                normalize_embedding=True,
            )
            state_features = full_state["handcrafted"]
            state_embedding = full_state["embedding"]

            diff_info = estimate_difficulty(state_features)
            difficulty_level = diff_info["difficulty_level"]
            difficulty_hist[difficulty_level] += 1

            scores = compute_action_scores(
                model=model,
                state_features=state_features,
                state_embedding=state_embedding,
                action_space=action_space,
                device=device,
            )

            chosen_action_idx = int(torch.argmax(scores).item())
            chosen_action = action_space.describe_action(chosen_action_idx)

            result = controller.execute(
                question=question,
                gold_answer=gold,
                action_idx=chosen_action_idx,
            )

            correct = result.extracted_answer == gold
            reward = result.reward_breakdown["total_reward"]

            total_correct += int(correct)
            total_reward += reward
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_total_tokens += result.total_tokens

            action_hist[str(chosen_action_idx)] = action_hist.get(str(chosen_action_idx), 0) + 1

            diff_bucket = action_by_difficulty[difficulty_level]
            diff_bucket[str(chosen_action_idx)] = diff_bucket.get(str(chosen_action_idx), 0) + 1

            records.append(
                {
                    "sample_idx": global_idx,
                    "question": question,
                    "gold": gold,
                    "difficulty": diff_info,
                    "state_features": state_features,
                    "chosen_action_idx": chosen_action_idx,
                    "chosen_action": chosen_action,
                    "chosen_action_bucket": bucket_key(chosen_action),
                    "best_score": float(scores[chosen_action_idx].item()),
                    "all_scores": [float(x.item()) for x in scores],
                    "pred": result.extracted_answer,
                    "correct": correct,
                    "model_name": result.model_name,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "verification_used": result.verification_used,
                    "format_ok": result.format_ok,
                    "reward_breakdown": result.reward_breakdown,
                }
            )

            print(
                f"[{global_idx}] "
                f"difficulty={difficulty_level} "
                f"action={chosen_action_idx} "
                f"correct={correct} "
                f"reward={reward:.4f}"
            )

    n = len(dataset)

    summary = {
        "model_path": model_path,
        "split": args.split,
        "start_idx": args.start_idx,
        "num_samples": n,
        "accuracy": total_correct / max(n, 1),
        "avg_reward": total_reward / max(n, 1),
        "avg_prompt_tokens": total_prompt_tokens / max(n, 1),
        "avg_completion_tokens": total_completion_tokens / max(n, 1),
        "avg_total_tokens": total_total_tokens / max(n, 1),
        "difficulty_hist": difficulty_hist,
        "chosen_action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        "action_by_difficulty": {
            level: dict(sorted(hist.items(), key=lambda x: x[1], reverse=True))
            for level, hist in action_by_difficulty.items()
        },
    }

    output = {
        "summary": summary,
        "records": records,
    }

    print("\n[SUMMARY]")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.save_json:
        parent = os.path.dirname(args.save_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] saved batch eval json -> {args.save_json}")

    if args.save_records:
        parent = os.path.dirname(args.save_records)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_records, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[INFO] saved records json -> {args.save_records}")


if __name__ == "__main__":
    main()