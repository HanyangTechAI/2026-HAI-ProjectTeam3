from typing import Dict, List

import torch

from src.preference.action_encoder import encode_action_features


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


def compute_model_scores(
    model,
    state_features: dict,
    state_embedding: list,
    action_space,
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


def select_topk_actions(scores: torch.Tensor, k: int) -> List[int]:
    k = max(1, min(k, scores.shape[0]))
    topk = torch.argsort(scores, descending=True)[:k]
    return [int(x.item()) for x in topk]


def rerank_topk_actions(
    baseline_scores: torch.Tensor,
    reranker_scores: torch.Tensor,
    topk: int,
    alpha: float = 0.7,
) -> Dict:
    """
    alpha:
        baseline importance.
        final_score = alpha * baseline + (1 - alpha) * reranker
    """
    candidate_indices = select_topk_actions(baseline_scores, topk)

    rows = []
    for idx in candidate_indices:
        b = float(baseline_scores[idx].item())
        r = float(reranker_scores[idx].item())
        final_score = alpha * b + (1.0 - alpha) * r
        rows.append(
            {
                "action_idx": idx,
                "baseline_score": b,
                "reranker_score": r,
                "final_score": final_score,
            }
        )

    rows = sorted(rows, key=lambda x: x["final_score"], reverse=True)
    chosen = rows[0]

    return {
        "chosen_action_idx": chosen["action_idx"],
        "candidates": rows,
        "topk": topk,
        "alpha": alpha,
    }