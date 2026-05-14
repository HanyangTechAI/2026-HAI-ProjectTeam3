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
from src.llm_client import build_llm_client
from src.preference.preference_model import ActionPreferenceNet
from src.rl.reranker import compute_model_scores, rerank_topk_actions


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_baseline_model_path(cfg: TrainConfig) -> str:
    split = getattr(cfg, "oracle_split", "train")
    start_idx = getattr(cfg, "oracle_start_idx", 0)
    num_samples = getattr(cfg, "oracle_num_samples", 50)
    suffix = f"{split}_{start_idx}_{start_idx + num_samples - 1}"

    candidate_patterns = [
        os.path.join(cfg.output_dir, f"action_preference_model_cost_aware_{suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_balanced_{suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_hard_{suffix}*.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_{suffix}*.pt"),
    ]

    matches = []
    for pattern in candidate_patterns:
        matches.extend(glob.glob(pattern))

    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError("No baseline preference model checkpoint found.")

    priority = [
        "action_preference_model_cost_aware_",
        "action_preference_model_balanced_",
        "action_preference_model_hard_",
        "action_preference_model_",
    ]

    def key_fn(path: str):
        name = os.path.basename(path)
        for i, p in enumerate(priority):
            if name.startswith(p):
                return (i, len(name))
        return (999, len(name))

    matches.sort(key=key_fn)
    return matches[0]


def resolve_rl_model_path(cfg: TrainConfig) -> Optional[str]:
    candidate_paths = [
        os.path.join(cfg.output_dir, "action_preference_model_rl_best.pt"),
        os.path.join(cfg.output_dir, "action_preference_model_rl_latest.pt"),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", type=str, required=True)
    parser.add_argument("--gold", type=str, default="")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--alpha", type=float, default=0.7)
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    baseline_path = resolve_baseline_model_path(cfg)
    rl_path = resolve_rl_model_path(cfg)

    baseline_model, _ = load_model(baseline_path, device=device)

    # RL model 없으면 baseline을 reranker로도 사용
    if rl_path is not None:
        reranker_model, _ = load_model(rl_path, device=device)
    else:
        reranker_model = baseline_model

    question = args.question.strip()
    gold = args.gold.strip()

    full_state = build_full_state(
        question=question,
        model_name=cfg.embedding_model_name,
        normalize_embedding=True,
    )
    state_features = full_state["handcrafted"]
    state_embedding = full_state["embedding"]

    difficulty = estimate_difficulty(state_features)

    action_space = InferenceActionSpace()

    with torch.no_grad():
        baseline_scores = compute_model_scores(
            model=baseline_model,
            state_features=state_features,
            state_embedding=state_embedding,
            action_space=action_space,
            device=device,
        )

        reranker_scores = compute_model_scores(
            model=reranker_model,
            state_features=state_features,
            state_embedding=state_embedding,
            action_space=action_space,
            device=device,
        )

    rerank_out = rerank_topk_actions(
        baseline_scores=baseline_scores,
        reranker_scores=reranker_scores,
        topk=args.topk,
        alpha=args.alpha,
    )

    chosen_action_idx = rerank_out["chosen_action_idx"]

    llm_client = build_llm_client(cfg.api_mode)
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    result = controller.execute(
        question=question,
        gold_answer=gold,
        action_idx=chosen_action_idx,
    )

    output = {
        "question": question,
        "gold": gold if gold != "" else None,
        "difficulty": difficulty,
        "baseline_model_path": baseline_path,
        "reranker_model_path": rl_path if rl_path is not None else baseline_path,
        "topk": args.topk,
        "alpha": args.alpha,
        "baseline_topk_rerank": rerank_out,
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": result.action_description,
        "execution_result": {
            "model_name": result.model_name,
            "pred": result.extracted_answer,
            "correct": (result.extracted_answer == gold) if gold != "" else None,
            "raw_text": result.raw_text,
            "final_text": result.final_text,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "verification_used": result.verification_used,
            "format_ok": result.format_ok,
            "reward_breakdown": result.reward_breakdown if gold != "" else None,
        },
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()