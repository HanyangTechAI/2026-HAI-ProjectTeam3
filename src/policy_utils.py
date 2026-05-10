import glob
import os
from typing import Dict, List, Optional

import torch

from src.controller.action_space import InferenceActionSpace
from src.preference.action_encoder import encode_action_features
from src.preference.preference_model import ActionPreferenceNet


HANDCRAFTED_FEATURE_ORDER = [
    "normalized_length",
    "normalized_word_count",
    "normalized_digit_count",
    "has_percent",
    "has_money",
    "has_ratio_words",
    "has_multistep_hint",
]


def build_state_vector(state_features: dict, state_embedding: list) -> List[float]:
    return [state_features[name] for name in HANDCRAFTED_FEATURE_ORDER] + state_embedding


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
    state_x = build_state_tensor(state_features, state_embedding, device=device)
    scores = []

    for action_idx in range(len(action_space)):
        action_vec = encode_action_features(action_idx, action_space)
        action_x = torch.tensor([action_vec], dtype=torch.float32, device=device)
        scores.append(model(state_x, action_x).squeeze())

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


def resolve_policy_checkpoint(output_dir: str, preferred_path: Optional[str] = None) -> str:
    if preferred_path:
        if not os.path.exists(preferred_path):
            raise FileNotFoundError(f"Checkpoint not found: {preferred_path}")
        return preferred_path

    candidate_paths = [
        os.path.join(output_dir, "action_preference_model_rl_best.pt"),
        os.path.join(output_dir, "action_preference_model_hard_train_0_49.pt"),
        os.path.join(output_dir, "action_preference_model_balanced_train_0_49.pt"),
        os.path.join(output_dir, "action_preference_model_cost_aware_train_0_49.pt"),
        os.path.join(output_dir, "action_preference_model.pt"),
    ]

    for path in candidate_paths:
        if os.path.exists(path):
            return path

    matches = sorted(glob.glob(os.path.join(output_dir, "action_preference_model*.pt")))
    if matches:
        return matches[0]

    raise FileNotFoundError(f"No action preference checkpoint found in {output_dir}")


def load_preference_model(model_path: str, device: str):
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "state_dim", "action_dim"}
    missing = required - set(checkpoint.keys())
    if missing:
        raise ValueError(f"Checkpoint missing required keys: {sorted(missing)}")

    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=int(checkpoint.get("hidden_dim", 128)),
        dropout=float(checkpoint.get("dropout", 0.1)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint
