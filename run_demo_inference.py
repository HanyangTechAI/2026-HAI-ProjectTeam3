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

    raise FileNotFoundError("No preference model checkpoint found.")


def estimate_difficulty(state_features: Dict[str, float]) -> Dict[str, float]:
    """
    발표/시연용 난이도 추정 점수.
    policy 입력과는 별개로 설명 가능성을 높이기 위한 heuristic이다.
    """
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


def explain_state(state_features: Dict[str, float]) -> List[str]:
    reasons = []

    if state_features["normalized_word_count"] >= 0.5:
        reasons.append("문장 길이가 비교적 길다")
    if state_features["normalized_digit_count"] >= 0.2:
        reasons.append("숫자 정보가 많다")
    if state_features["has_ratio_words"] > 0:
        reasons.append("배수·비율 표현이 포함되어 있다")
    if state_features["has_multistep_hint"] > 0:
        reasons.append("다단계 추론 힌트가 있다")
    if state_features["has_money"] > 0:
        reasons.append("금액 계산 문제다")
    if state_features["has_percent"] > 0:
        reasons.append("퍼센트 개념이 포함되어 있다")

    if not reasons:
        reasons.append("문제 구조가 비교적 단순하다")

    return reasons


def explain_action(action_desc: Dict) -> str:
    reasoning = action_desc["reasoning_budget"]
    model_route = action_desc["model_route"]
    verify = action_desc["verify"]

    parts = []

    if reasoning == "none":
        parts.append("짧고 직접적인 풀이 전략")
    elif reasoning == "short":
        parts.append("가벼운 추론을 사용하는 전략")
    elif reasoning == "long":
        parts.append("더 깊은 추론을 사용하는 전략")

    if model_route == "small":
        parts.append("비용이 낮은 소형 모델 사용")
    elif model_route == "large":
        parts.append("정확도 우선의 대형 모델 사용")

    if verify:
        parts.append("출력 형식 검증 포함")
    else:
        parts.append("추가 검증 없이 바로 출력")

    return ", ".join(parts)


def build_policy_explanation(
    question: str,
    state_features: Dict[str, float],
    chosen_action: Dict,
    top_actions: List[Dict],
) -> Dict:
    difficulty_info = estimate_difficulty(state_features)
    state_reasons = explain_state(state_features)
    action_reason = explain_action(chosen_action)

    summary = (
        f"이 문제는 {difficulty_info['difficulty_level']} 난이도로 추정되며 "
        f"(score={difficulty_info['difficulty_score']:.3f}), "
        f"현재 정책은 action {chosen_action['action_idx']}를 선택했다. "
        f"선택 전략은 '{action_reason}'이다."
    )

    return {
        "question": question,
        "difficulty": difficulty_info,
        "state_reasons": state_reasons,
        "chosen_action_reason": action_reason,
        "summary": summary,
        "top_action_candidates": top_actions,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Input question for policy demo inference.",
    )
    parser.add_argument(
        "--gold",
        type=str,
        default="",
        help="Optional gold answer for reward/correctness check.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="How many top action candidates to show.",
    )
    args = parser.parse_args()

    cfg = TrainConfig()
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = resolve_model_path(cfg)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    print(checkpoint.keys())
    print(checkpoint["state_dim"], checkpoint["action_dim"])
    print(checkpoint["model_state_dict"].keys())

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

    with torch.no_grad():
        scores = compute_action_scores(
            model=model,
            state_features=state_features,
            state_embedding=state_embedding,
            action_space=action_space,
            device=device,
        )

    sorted_indices = torch.argsort(scores, descending=True).tolist()
    chosen_action_idx = int(sorted_indices[0])

    topk = max(1, min(args.topk, len(sorted_indices)))
    top_actions = []
    for idx in sorted_indices[:topk]:
        top_actions.append(
            {
                "action_idx": int(idx),
                "action": action_space.describe_action(int(idx)),
                "score": float(scores[int(idx)].item()),
            }
        )

    result = controller.execute(
        question=question,
        gold_answer=gold,
        action_idx=chosen_action_idx,
    )

    explanation = build_policy_explanation(
        question=question,
        state_features=state_features,
        chosen_action=result.action_description,
        top_actions=top_actions,
    )

    output = {
        "model_path": model_path,
        "question": question,
        "state_features": state_features,
        "chosen_action_idx": chosen_action_idx,
        "chosen_action": result.action_description,
        "all_action_scores": {
            str(i): float(scores[i].item()) for i in range(len(action_space))
        },
        "top_action_candidates": top_actions,
        "policy_explanation": explanation,
        "execution_result": {
            "model_name": result.model_name,
            "prompt": result.prompt,
            "raw_text": result.raw_text,
            "final_text": result.final_text,
            "pred": result.extracted_answer,
            "gold": gold,
            "correct": (result.extracted_answer == gold) if gold != "" else None,
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