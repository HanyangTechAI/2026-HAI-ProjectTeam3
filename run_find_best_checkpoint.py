import argparse
import glob
import json
import os
import random
from typing import Dict, List, Optional

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


def find_all_pt_files(output_dir: str) -> List[str]:
    pattern = os.path.join(output_dir, "*.pt")
    paths = glob.glob(pattern)
    return sorted(set(paths))


def safe_load_checkpoint(model_path: str, device: str) -> Dict:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dict.")

    required_keys = {"model_state_dict", "state_dim", "action_dim"}
    missing = required_keys - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")

    return checkpoint


def safe_build_model(checkpoint: Dict, device: str):
    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_checkpoint(
    model_path: str,
    cfg: TrainConfig,
    dataset,
    action_space: InferenceActionSpace,
    controller: AdaptiveInferenceController,
    device: str,
) -> Dict:
    checkpoint = safe_load_checkpoint(model_path, device)
    model = safe_build_model(checkpoint, device)

    total_correct = 0
    total_reward = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0

    chosen_action_hist = {}
    difficulty_hist = {"easy": 0, "medium": 0, "hard": 0}
    action_by_difficulty = {
        "easy": {},
        "medium": {},
        "hard": {},
    }

    with torch.no_grad():
        for sample in dataset:
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

            key = str(chosen_action_idx)
            chosen_action_hist[key] = chosen_action_hist.get(key, 0) + 1

            bucket = action_by_difficulty[difficulty_level]
            bucket[key] = bucket.get(key, 0) + 1

    n = len(dataset)

    return {
        "model_path": model_path,
        "filename": os.path.basename(model_path),
        "data_suffix": checkpoint.get("data_suffix", None),
        "state_dim": checkpoint.get("state_dim"),
        "action_dim": checkpoint.get("action_dim"),
        "num_samples": n,
        "accuracy": total_correct / max(n, 1),
        "avg_reward": total_reward / max(n, 1),
        "avg_prompt_tokens": total_prompt_tokens / max(n, 1),
        "avg_completion_tokens": total_completion_tokens / max(n, 1),
        "avg_total_tokens": total_total_tokens / max(n, 1),
        "chosen_action_hist": dict(sorted(chosen_action_hist.items(), key=lambda x: x[1], reverse=True)),
        "difficulty_hist": difficulty_hist,
        "action_by_difficulty": {
            level: dict(sorted(hist.items(), key=lambda x: x[1], reverse=True))
            for level, hist in action_by_difficulty.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--save_json", type=str, default="outputs/checkpoint_comparison_all_pt.json")
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=args.split,
        n_samples=args.start_idx + args.num_samples,
    )
    dataset = dataset.select(range(args.start_idx, args.start_idx + args.num_samples))

    action_space = InferenceActionSpace()
    llm_client = build_llm_client(cfg.api_mode)
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    candidates = find_all_pt_files(cfg.output_dir)
    if not candidates:
        raise FileNotFoundError(f"No .pt files found in {cfg.output_dir}")

    print(f"[INFO] found {len(candidates)} .pt files in {cfg.output_dir}")

    results = []
    for idx, model_path in enumerate(candidates, start=1):
        print(f"\n[{idx}/{len(candidates)}] evaluating: {model_path}")

        try:
            summary = evaluate_checkpoint(
                model_path=model_path,
                cfg=cfg,
                dataset=dataset,
                action_space=action_space,
                controller=controller,
                device=device,
            )
            results.append(summary)
            print(
                f"  accuracy={summary['accuracy']:.4f} "
                f"avg_reward={summary['avg_reward']:.4f} "
                f"hist={summary['chosen_action_hist']}"
            )
        except Exception as e:
            error_row = {
                "model_path": model_path,
                "filename": os.path.basename(model_path),
                "error": str(e),
            }
            results.append(error_row)
            print(f"  [SKIP] {e}")

    valid_results = [r for r in results if "error" not in r]
    valid_results = sorted(
        valid_results,
        key=lambda x: (x["accuracy"], x["avg_reward"]),
        reverse=True,
    )

    output = {
        "split": args.split,
        "start_idx": args.start_idx,
        "num_samples": args.num_samples,
        "num_candidates": len(candidates),
        "num_valid": len(valid_results),
        "num_skipped": len(results) - len(valid_results),
        "best_model": valid_results[0] if valid_results else None,
        "ranking": valid_results,
        "raw_results": results,
    }

    if args.save_json:
        parent = os.path.dirname(args.save_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

    print("\n[TOP VALID MODELS]")
    for i, row in enumerate(valid_results[:10], start=1):
        print(
            f"{i}. {row['filename']} | "
            f"acc={row['accuracy']:.4f} | "
            f"reward={row['avg_reward']:.4f} | "
            f"hist={row['chosen_action_hist']}"
        )

    if valid_results:
        print("\n[BEST MODEL]")
        print(json.dumps(valid_results[0], ensure_ascii=False, indent=2))

    if args.save_json:
        print(f"\n[INFO] saved comparison -> {args.save_json}")


if __name__ == "__main__":
    main()