import argparse
import os
import random

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
from src.imitation.dataset_builder import save_json
from src.policy_utils import resolve_policy_checkpoint


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_state_vector(state_features: dict, state_embedding: list):
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


def build_state_tensor(state_features: dict, state_embedding: list, device: str):
    vec = build_state_vector(state_features, state_embedding)
    return torch.tensor([vec], dtype=torch.float32, device=device)


def compute_action_scores(
    model: ActionPreferenceNet,
    state_features: dict,
    state_embedding: list,
    action_space: InferenceActionSpace,
    device: str,
):
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


def resolve_model_path(cfg: TrainConfig, preferred_path: str = ""):
    if preferred_path:
        return resolve_policy_checkpoint(cfg.output_dir, preferred_path)

    train_split = getattr(cfg, "oracle_split", "train")
    train_start_idx = getattr(cfg, "oracle_start_idx", 0)
    train_num_samples = getattr(cfg, "oracle_num_samples", 50)
    train_suffix = f"{train_split}_{train_start_idx}_{train_start_idx + train_num_samples - 1}"

    candidate_paths = [
        os.path.join(cfg.output_dir, f"action_preference_model_balanced_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_combined_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_cost_aware_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_hard_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, "action_preference_model.pt"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    raise FileNotFoundError("No preference model checkpoint found.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="")
    parser.add_argument("--num_samples", type=int, default=0)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--api_mode", type=str, default="")
    parser.add_argument("--embedding_model", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    args = parser.parse_args()

    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    eval_split = args.split or getattr(cfg, "test_split", "test")
    eval_samples = args.num_samples or getattr(cfg, "test_samples", 30)
    embedding_model = args.embedding_model or cfg.embedding_model_name
    api_mode = args.api_mode or cfg.api_mode

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=eval_split,
        n_samples=args.start_idx + eval_samples,
    )
    dataset_end_idx = min(len(dataset), args.start_idx + eval_samples)
    dataset = dataset.select(range(args.start_idx, dataset_end_idx))

    model_path = resolve_model_path(cfg, preferred_path=args.checkpoint)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    llm_client = build_llm_client(api_mode)
    action_space = InferenceActionSpace()
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    total_correct = 0
    total_reward = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_total_tokens = 0
    chosen_action_hist = {}
    records = []

    with torch.no_grad():
        for idx, sample in enumerate(dataset):
            global_idx = args.start_idx + idx
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            full_state = build_full_state(
                question=question,
                model_name=embedding_model,
                normalize_embedding=True,
            )
            state_features = full_state["handcrafted"]
            state_embedding = full_state["embedding"]

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
            chosen_action_hist[chosen_action_idx] = chosen_action_hist.get(chosen_action_idx, 0) + 1

            records.append(
                {
                    "eval_sample_idx": idx,
                    "sample_idx": global_idx,
                    "question": question,
                    "gold": gold,
                    "chosen_action_idx": chosen_action_idx,
                    "chosen_action": result.action_description,
                    "best_score": float(scores[chosen_action_idx].item()),
                    "scores": [float(x.item()) for x in scores],
                    "pred": result.extracted_answer,
                    "correct": correct,
                    "model_name": result.model_name,
                    "raw_text": result.raw_text,
                    "final_text": result.final_text,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "verification_used": result.verification_used,
                    "format_ok": result.format_ok,
                    "reward_breakdown": result.reward_breakdown,
                }
            )

    n = len(dataset)
    summary = {
        "split": eval_split,
        "start_idx": args.start_idx,
        "num_samples": n,
        "api_mode": api_mode,
        "embedding_model": embedding_model,
        "accuracy": total_correct / max(n, 1),
        "avg_reward": total_reward / max(n, 1),
        "avg_prompt_tokens": total_prompt_tokens / max(n, 1),
        "avg_completion_tokens": total_completion_tokens / max(n, 1),
        "avg_total_tokens": total_total_tokens / max(n, 1),
        "chosen_action_hist": {
            str(k): v for k, v in sorted(chosen_action_hist.items(), key=lambda x: x[1], reverse=True)
        },
        "model_path": model_path,
    }

    end_idx = args.start_idx + n - 1
    summary_path = os.path.join(cfg.output_dir, f"preference_holdout_summary_{eval_split}_{args.start_idx}_{end_idx}.json")
    records_path = os.path.join(cfg.output_dir, f"preference_holdout_records_{eval_split}_{args.start_idx}_{end_idx}.json")

    save_json(summary_path, summary)
    save_json(records_path, records)

    print("[HOLDOUT SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()
