import json
import os

import torch

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import load_gsm8k_subset, extract_gold_answer
from archives.grpo.policy_model import GRPOPolicyNet
from src.llm_client import build_llm_client


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_state_tensor(state_features: dict, state_embedding: list, device: str = "cpu") -> torch.Tensor:
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
def choose_action_argmax(model, state_features, state_embedding, device: str):
    state_x = build_state_tensor(state_features, state_embedding, device=device)
    logits = model(state_x)
    probs = torch.softmax(logits, dim=-1).squeeze(0)

    best_action = int(torch.argmax(probs).item())
    return {
        "chosen_action_idx": best_action,
        "probs": probs.detach().cpu().tolist(),
        "confidence": float(probs[best_action].item()),
    }


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

    model_path = os.path.join(cfg.output_dir, "grpo_style_policy.pt")
    checkpoint = torch.load(model_path, map_location=device)

    model = GRPOPolicyNet(
        input_dim=checkpoint["input_dim"],
        hidden_dim=128,
        num_actions=checkpoint["num_actions"],
        dropout=0.1,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0

    action_hist = {}
    records = []

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

        pol = choose_action_argmax(
            model=model,
            state_features=state_features,
            state_embedding=state_embedding,
            device=device,
        )

        chosen_action_idx = pol["chosen_action_idx"]

        result = controller.execute(
            question=question,
            gold_answer=gold,
            action_idx=chosen_action_idx,
        )

        correct = (result.extracted_answer == gold)
        reward = result.reward_breakdown["total_reward"]

        total_reward += reward
        total_correct += int(correct)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_tokens += result.total_tokens
        action_hist[chosen_action_idx] = action_hist.get(chosen_action_idx, 0) + 1

        records.append(
            {
                "sample_idx": sample_idx,
                "question": question,
                "gold": gold,
                "chosen_action_idx": chosen_action_idx,
                "chosen_action": result.action_description,
                "confidence": pol["confidence"],
                "probs": pol["probs"],
                "model_name": result.model_name,
                "raw_text": result.raw_text,
                "final_text": result.final_text,
                "pred": result.extracted_answer,
                "correct": correct,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "verification_used": result.verification_used,
                "format_ok": result.format_ok,
                "reward_breakdown": result.reward_breakdown,
            }
        )

        print(
            f"[SAMPLE {sample_idx}] "
            f"action={chosen_action_idx} "
            f"conf={pol['confidence']:.4f} "
            f"correct={correct} "
            f"reward={reward:.4f}"
        )

    n = len(dataset)
    summary = {
        "num_samples": n,
        "accuracy": total_correct / n,
        "avg_reward": total_reward / n,
        "avg_prompt_tokens": total_prompt_tokens / n,
        "avg_completion_tokens": total_completion_tokens / n,
        "avg_total_tokens": total_tokens / n,
        "chosen_action_hist": {
            str(k): v for k, v in sorted(action_hist.items(), key=lambda x: x[1], reverse=True)
        },
    }

    save_json(os.path.join(cfg.output_dir, "grpo_controller_records.json"), records)
    save_json(os.path.join(cfg.output_dir, "grpo_controller_summary.json"), summary)

    print("\n[SUMMARY]")
    print(summary)


if __name__ == "__main__":
    main()