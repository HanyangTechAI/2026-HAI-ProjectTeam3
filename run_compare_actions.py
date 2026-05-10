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

    # fallback: loose search
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


def explain_action(action_desc: Dict) -> str:
    reasoning = action_desc["reasoning_budget"]
    model_route = action_desc["model_route"]
    verify = action_desc["verify"]

    parts = []

    if reasoning == "none":
        parts.append("직접 출력")
    elif reasoning == "short":
        parts.append("짧은 내부 추론")
    elif reasoning == "long":
        parts.append("긴 내부 추론")

    if model_route == "small":
        parts.append("소형 모델")
    elif model_route == "large":
        parts.append("대형 모델")

    if verify:
        parts.append("검증 포함")
    else:
        parts.append("검증 없음")

    return ", ".join(parts)


def summarize_state_reasons(state_features: Dict[str, float]) -> List[str]:
    reasons = []

    if state_features["normalized_word_count"] >= 0.5:
        reasons.append("문장 길이가 비교적 길다")
    if state_features["normalized_digit_count"] >= 0.2:
        reasons.append("숫자 정보가 많다")
    if state_features["has_ratio_words"] > 0:
        reasons.append("배수/비율 표현이 있다")
    if state_features["has_multistep_hint"] > 0:
        reasons.append("다단계 추론 힌트가 있다")
    if state_features["has_money"] > 0:
        reasons.append("금액 계산이 포함된다")
    if state_features["has_percent"] > 0:
        reasons.append("퍼센트 개념이 포함된다")

    if not reasons:
        reasons.append("문제 구조가 단순한 편이다")

    return reasons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Input question to compare all actions.",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default="",
        help="Optional gold answer for correctness/reward comparison.",
    )
    parser.add_argument(
        "--save_json",
        type=str,
        default="",
        help="Optional output json path.",
    )
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
    state_reasons = summarize_state_reasons(state_features)

    with torch.no_grad():
        policy_scores = compute_action_scores(
            model=model,
            state_features=state_features,
            state_embedding=state_embedding,
            action_space=action_space,
            device=device,
        )

    results = []

    for action_idx in range(len(action_space)):
        action_desc = action_space.describe_action(action_idx)
        result = controller.execute(
            question=question,
            gold_answer=gold,
            action_idx=action_idx,
        )

        pred = result.extracted_answer
        correct = (pred == gold) if gold != "" else None
        reward = result.reward_breakdown["total_reward"] if gold != "" else None

        results.append(
            {
                "action_idx": action_idx,
                "action": action_desc,
                "action_explanation": explain_action(action_desc),
                "policy_score": float(policy_scores[action_idx].item()),
                "model_name": result.model_name,
                "pred": pred,
                "gold": gold if gold != "" else None,
                "correct": correct,
                "raw_text": result.raw_text,
                "final_text": result.final_text,
                "prompt": result.prompt,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "verification_used": result.verification_used,
                "format_ok": result.format_ok,
                "reward_breakdown": result.reward_breakdown if gold != "" else None,
                "reward": reward,
            }
        )

    # policy best
    policy_best_idx = int(torch.argmax(policy_scores).item())
    policy_best = next(r for r in results if r["action_idx"] == policy_best_idx)

    # reward best if gold exists
    reward_best = None
    if gold != "":
        reward_best = max(results, key=lambda x: x["reward"])

    # sort for presentation
    if gold != "":
        results_sorted = sorted(
            results,
            key=lambda x: (x["reward"], x["policy_score"]),
            reverse=True,
        )
    else:
        results_sorted = sorted(
            results,
            key=lambda x: x["policy_score"],
            reverse=True,
        )

    output = {
        "model_path": model_path,
        "question": question,
        "gold": gold if gold != "" else None,
        "difficulty": difficulty,
        "state_features": state_features,
        "state_reasons": state_reasons,
        "policy_best_action_idx": policy_best_idx,
        "policy_best_action": policy_best["action"],
        "policy_best_action_explanation": policy_best["action_explanation"],
        "reward_best_action_idx": reward_best["action_idx"] if reward_best is not None else None,
        "reward_best_action": reward_best["action"] if reward_best is not None else None,
        "results": results_sorted,
    }

    print(json.dumps(output, ensure_ascii=False, indent=2))

    if args.save_json:
        parent = os.path.dirname(args.save_json)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        print(f"\n[INFO] saved comparison json -> {args.save_json}")


if __name__ == "__main__":
    main()