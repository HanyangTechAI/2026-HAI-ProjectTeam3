import json
import os

import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import load_gsm8k_subset, extract_gold_answer
from archives.policy_model import ImitationPolicyNet
from src.llm_client import build_llm_client


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_full_input(state_features: dict, state_embedding: list, device: str = "cpu") -> torch.Tensor:
    handcrafted = [
        state_features["normalized_length"],
        state_features["normalized_word_count"],
        state_features["normalized_digit_count"],
        state_features["has_percent"],
        state_features["has_money"],
        state_features["has_ratio_words"],
        state_features["has_multistep_hint"],
    ]
    x = handcrafted + state_embedding
    return torch.tensor([x], dtype=torch.float32, device=device)


@torch.no_grad()
def get_topk_actions(
    model,
    state_features: dict,
    state_embedding: list,
    k: int,
    device: str,
):
    x = build_full_input(state_features, state_embedding, device=device)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    topk_probs, topk_indices = torch.topk(probs, k=k)

    candidates = []
    for prob, idx in zip(topk_probs.tolist(), topk_indices.tolist()):
        candidates.append(
            {
                "action_idx": int(idx),
                "policy_prob": float(prob),
            }
        )
    return candidates, probs.tolist()


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=cfg.test_split,
        n_samples=cfg.test_samples,
    )

    llm_client = build_llm_client(cfg.api_mode)
    action_space = InferenceActionSpace()
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    model_path = os.path.join(cfg.output_dir, "imitation_policy_embedding.pt")
    checkpoint = torch.load(model_path, map_location=device)

    model = ImitationPolicyNet(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_actions=checkpoint["num_actions"],
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    top_k = 3

    rollout_records = []
    improved_dataset = []
    preference_pairs = []

    for sample_idx, sample in enumerate(dataset):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        full_state = build_full_state(
            question=question,
            model_name=cfg.embedding_model_name,
            normalize_embedding=True,
        )
        state_features = full_state["handcrafted"]
        state_embedding = full_state["embedding"]

        candidates, full_probs = get_topk_actions(
            model=model,
            state_features=state_features,
            state_embedding=state_embedding,
            k=top_k,
            device=device,
        )

        executed = []
        for cand in candidates:
            action_idx = cand["action_idx"]
            result = controller.execute(
                question=question,
                gold_answer=gold,
                action_idx=action_idx,
            )

            executed.append(
                {
                    "action_idx": action_idx,
                    "policy_prob": cand["policy_prob"],
                    "action": result.action_description,
                    "pred": result.extracted_answer,
                    "correct": result.extracted_answer == gold,
                    "reward": result.reward_breakdown["total_reward"],
                    "reward_breakdown": result.reward_breakdown,
                    "model_name": result.model_name,
                    "raw_text": result.raw_text,
                    "final_text": result.final_text,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "verification_used": result.verification_used,
                    "format_ok": result.format_ok,
                }
            )

        executed.sort(key=lambda x: x["reward"], reverse=True)
        best = executed[0]

        rollout_records.append(
            {
                "sample_idx": sample_idx,
                "question": question,
                "gold": gold,
                "state_features": state_features,
                "state_embedding": state_embedding,
                "full_policy_probs": full_probs,
                "candidates": executed,
                "best_candidate": best,
            }
        )

        improved_dataset.append(
            {
                "question": question,
                "state_features": state_features,
                "state_embedding": state_embedding,
                "label_action_idx": best["action_idx"],
                "gold": gold,
                "improved_reward": best["reward"],
                "improved_action": best["action"],
            }
        )

        # pairwise preference dataset
        for worse in executed[1:]:
            preference_pairs.append(
                {
                    "question": question,
                    "state_features": state_features,
                    "state_embedding": state_embedding,
                    "preferred_action_idx": best["action_idx"],
                    "rejected_action_idx": worse["action_idx"],
                    "preferred_reward": best["reward"],
                    "rejected_reward": worse["reward"],
                }
            )

        print(
            f"[SAMPLE {sample_idx}] "
            f"best_action={best['action_idx']} "
            f"best_reward={best['reward']:.4f}"
        )

    save_json(
        os.path.join(cfg.output_dir, "policy_improvement_rollout_records.json"),
        rollout_records,
    )
    save_json(
        os.path.join(cfg.output_dir, "policy_improvement_dataset.json"),
        improved_dataset,
    )
    save_json(
        os.path.join(cfg.output_dir, "policy_preference_pairs.json"),
        preference_pairs,
    )

    print(f"[INFO] saved rollout records")
    print(f"[INFO] saved improved dataset: {len(improved_dataset)} samples")
    print(f"[INFO] saved preference pairs: {len(preference_pairs)} pairs")


if __name__ == "__main__":
    main()