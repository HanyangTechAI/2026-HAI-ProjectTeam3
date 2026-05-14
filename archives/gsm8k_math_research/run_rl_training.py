import copy
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from configs import TrainConfig
from src.controller.action_space import InferenceActionSpace
from src.controller.runtime_controller import AdaptiveInferenceController
from src.controller.state_encoder import build_full_state
from src.data import load_gsm8k_subset, extract_gold_answer
from src.llm_client import build_llm_client
from src.preference.action_encoder import encode_action_features
from src.preference.preference_model import ActionPreferenceNet


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


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


def scores_to_probs(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = scores / temperature
    return F.softmax(logits, dim=0)


def sample_action_from_scores(
    scores: torch.Tensor,
    temperature: float = 1.0,
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    probs = scores_to_probs(scores, temperature=temperature)
    dist = torch.distributions.Categorical(probs=probs)
    action = dist.sample()
    log_prob = dist.log_prob(action)
    return int(action.item()), log_prob, probs


def compute_entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs + 1e-12)).sum()


def compute_kl_divergence(current_probs: torch.Tensor, ref_probs: torch.Tensor) -> torch.Tensor:
    return (current_probs * (torch.log(current_probs + 1e-12) - torch.log(ref_probs + 1e-12))).sum()


class RewardNormalizer:
    def __init__(self, momentum: float = 0.95, eps: float = 1e-8):
        self.momentum = momentum
        self.eps = eps
        self.mean = 0.0
        self.var = 1.0
        self.initialized = False

    def normalize(self, reward: float) -> float:
        reward = float(reward)

        if not self.initialized:
            self.mean = reward
            self.var = 1.0
            self.initialized = True
            return 0.0

        old_mean = self.mean
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * reward
        centered = reward - old_mean
        self.var = self.momentum * self.var + (1.0 - self.momentum) * (centered ** 2)

        std = max(self.var, self.eps) ** 0.5
        return (reward - self.mean) / std

    def state_dict(self) -> Dict:
        return {
            "momentum": self.momentum,
            "eps": self.eps,
            "mean": self.mean,
            "var": self.var,
            "initialized": self.initialized,
        }


@torch.no_grad()
def evaluate_policy_greedy(
    model: ActionPreferenceNet,
    dataset,
    action_space: InferenceActionSpace,
    controller: AdaptiveInferenceController,
    cfg: TrainConfig,
    device: str,
    max_eval_samples: int,
    temperature: float,
) -> Dict:
    model.eval()

    total_reward = 0.0
    total_correct = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    action_hist = {}
    records = []

    eval_count = min(max_eval_samples, len(dataset))

    for local_idx, sample in enumerate(dataset.select(range(eval_count))):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        full_state = build_full_state(
            question=question,
            model_name=cfg.embedding_model_name,
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
        probs = scores_to_probs(scores, temperature=temperature)
        chosen_action_idx = int(torch.argmax(probs).item())

        result = controller.execute(
            question=question,
            gold_answer=gold,
            action_idx=chosen_action_idx,
        )

        correct = result.extracted_answer == gold
        reward = result.reward_breakdown["total_reward"]

        total_reward += reward
        total_correct += int(correct)
        total_prompt_tokens += result.prompt_tokens
        total_completion_tokens += result.completion_tokens
        total_tokens += result.total_tokens
        action_hist[chosen_action_idx] = action_hist.get(chosen_action_idx, 0) + 1

        records.append(
            {
                "eval_sample_idx": local_idx,
                "question": question,
                "gold": gold,
                "chosen_action_idx": chosen_action_idx,
                "chosen_action": result.action_description,
                "action_probs": probs.detach().cpu().tolist(),
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

    summary = {
        "num_samples": eval_count,
        "accuracy": total_correct / max(eval_count, 1),
        "avg_reward": total_reward / max(eval_count, 1),
        "avg_prompt_tokens": total_prompt_tokens / max(eval_count, 1),
        "avg_completion_tokens": total_completion_tokens / max(eval_count, 1),
        "avg_total_tokens": total_tokens / max(eval_count, 1),
        "chosen_action_hist": {
            str(k): v for k, v in sorted(action_hist.items(), key=lambda x: x[1], reverse=True)
        },
    }

    return {
        "summary": summary,
        "records": records,
    }


def get_base_model_path(cfg: TrainConfig) -> str:
    train_split = getattr(cfg, "oracle_split", "train")
    train_start_idx = getattr(cfg, "oracle_start_idx", 0)
    train_num_samples = getattr(cfg, "oracle_num_samples", 50)
    train_suffix = f"{train_split}_{train_start_idx}_{train_start_idx + train_num_samples - 1}"

    candidate_model_paths = [
        os.path.join(cfg.output_dir, f"action_preference_model_cost_aware_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_hard_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_balanced_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, f"action_preference_model_{train_suffix}.pt"),
        os.path.join(cfg.output_dir, "action_preference_model.pt"),
    ]

    for p in candidate_model_paths:
        if os.path.exists(p):
            return p

    raise FileNotFoundError(
        "No preference model checkpoint found. "
        "Please train a preference model first."
    )


def load_model_from_checkpoint(model_path: str, device: str) -> Tuple[ActionPreferenceNet, Dict]:
    checkpoint = torch.load(model_path, map_location=device)

    model = ActionPreferenceNet(
        state_dim=checkpoint["state_dim"],
        action_dim=checkpoint["action_dim"],
        hidden_dim=128,
        dropout=0.1,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    return model, checkpoint


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)
    set_seed(getattr(cfg, "seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"

    rl_train_split = getattr(cfg, "rl_train_split", "train")
    rl_train_samples = getattr(cfg, "rl_train_samples", 30)
    rl_eval_split = getattr(cfg, "rl_eval_split", "test")
    rl_eval_samples = getattr(cfg, "rl_eval_samples", 30)

    rl_epochs = getattr(cfg, "rl_epochs", 3)
    rl_lr = getattr(cfg, "rl_lr", 1e-5)
    rl_temperature = getattr(cfg, "rl_temperature", 1.0)

    rl_entropy_coef = getattr(cfg, "rl_entropy_coef", 0.01)
    rl_kl_coef = getattr(cfg, "rl_kl_coef", 0.05)
    rl_grad_clip_norm = getattr(cfg, "rl_grad_clip_norm", 1.0)
    rl_reward_norm_momentum = getattr(cfg, "rl_reward_norm_momentum", 0.95)

    train_ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=rl_train_split,
        n_samples=rl_train_samples,
    )
    eval_ds = load_gsm8k_subset(
        dataset_name=cfg.dataset_name,
        dataset_config=cfg.dataset_config,
        split=rl_eval_split,
        n_samples=rl_eval_samples,
    )

    llm_client = build_llm_client(cfg.api_mode)
    action_space = InferenceActionSpace()
    controller = AdaptiveInferenceController(
        llm_client=llm_client,
        action_space=action_space,
    )

    base_model_path = get_base_model_path(cfg)
    policy_model, checkpoint = load_model_from_checkpoint(base_model_path, device=device)

    reference_model = copy.deepcopy(policy_model).to(device)
    reference_model.eval()
    for p in reference_model.parameters():
        p.requires_grad = False

    optimizer = Adam(policy_model.parameters(), lr=rl_lr)
    reward_normalizer = RewardNormalizer(momentum=rl_reward_norm_momentum)

    print(f"[INFO] loaded base model from: {base_model_path}")
    print(f"[INFO] device: {device}")
    print(f"[INFO] train samples: {len(train_ds)}")
    print(f"[INFO] eval samples: {len(eval_ds)}")
    print(f"[INFO] num actions: {len(action_space)}")

    train_history = []
    best_eval_reward = -1e18
    best_eval_path = os.path.join(cfg.output_dir, "action_preference_model_rl_best.pt")
    latest_path = os.path.join(cfg.output_dir, "action_preference_model_rl_latest.pt")

    for epoch in range(1, rl_epochs + 1):
        policy_model.train()

        epoch_rewards = []
        epoch_norm_rewards = []
        epoch_losses = []
        epoch_policy_losses = []
        epoch_kl_losses = []
        epoch_entropies = []
        epoch_correct = 0
        epoch_action_hist = {}
        step_records = []

        for sample_idx, sample in enumerate(train_ds):
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            full_state = build_full_state(
                question=question,
                model_name=cfg.embedding_model_name,
                normalize_embedding=True,
            )
            state_features = full_state["handcrafted"]
            state_embedding = full_state["embedding"]

            current_scores = compute_action_scores(
                model=policy_model,
                state_features=state_features,
                state_embedding=state_embedding,
                action_space=action_space,
                device=device,
            )

            with torch.no_grad():
                ref_scores = compute_action_scores(
                    model=reference_model,
                    state_features=state_features,
                    state_embedding=state_embedding,
                    action_space=action_space,
                    device=device,
                )

            action_idx, log_prob, current_probs = sample_action_from_scores(
                scores=current_scores,
                temperature=rl_temperature,
            )
            ref_probs = scores_to_probs(ref_scores, temperature=rl_temperature)

            result = controller.execute(
                question=question,
                gold_answer=gold,
                action_idx=action_idx,
            )

            reward = float(result.reward_breakdown["total_reward"])
            norm_reward = reward_normalizer.normalize(reward)
            correct = int(result.extracted_answer == gold)

            entropy = compute_entropy(current_probs)
            kl = compute_kl_divergence(current_probs, ref_probs)

            policy_loss = -(log_prob * norm_reward)
            kl_loss = rl_kl_coef * kl
            entropy_bonus = rl_entropy_coef * entropy

            loss = policy_loss + kl_loss - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), rl_grad_clip_norm)
            optimizer.step()

            epoch_rewards.append(reward)
            epoch_norm_rewards.append(norm_reward)
            epoch_losses.append(float(loss.item()))
            epoch_policy_losses.append(float(policy_loss.item()))
            epoch_kl_losses.append(float(kl_loss.item()))
            epoch_entropies.append(float(entropy.item()))
            epoch_correct += correct
            epoch_action_hist[action_idx] = epoch_action_hist.get(action_idx, 0) + 1

            step_records.append(
                {
                    "epoch": epoch,
                    "step": sample_idx,
                    "question": question,
                    "gold": gold,
                    "action_idx": action_idx,
                    "reward": reward,
                    "normalized_reward": norm_reward,
                    "correct": bool(correct),
                    "loss": float(loss.item()),
                    "policy_loss": float(policy_loss.item()),
                    "kl_loss": float(kl_loss.item()),
                    "entropy": float(entropy.item()),
                    "action_probs": current_probs.detach().cpu().tolist(),
                    "ref_action_probs": ref_probs.detach().cpu().tolist(),
                    "reward_breakdown": result.reward_breakdown,
                }
            )

            print(
                f"[EPOCH {epoch}][STEP {sample_idx}] "
                f"action={action_idx} "
                f"correct={bool(correct)} "
                f"reward={reward:.4f} "
                f"norm_reward={norm_reward:.4f} "
                f"loss={loss.item():.4f}"
            )

        train_summary = {
            "epoch": epoch,
            "train_accuracy": epoch_correct / max(len(train_ds), 1),
            "train_avg_reward": float(np.mean(epoch_rewards)) if epoch_rewards else 0.0,
            "train_avg_norm_reward": float(np.mean(epoch_norm_rewards)) if epoch_norm_rewards else 0.0,
            "train_avg_loss": float(np.mean(epoch_losses)) if epoch_losses else 0.0,
            "train_avg_policy_loss": float(np.mean(epoch_policy_losses)) if epoch_policy_losses else 0.0,
            "train_avg_kl_loss": float(np.mean(epoch_kl_losses)) if epoch_kl_losses else 0.0,
            "train_avg_entropy": float(np.mean(epoch_entropies)) if epoch_entropies else 0.0,
            "train_action_hist": {
                str(k): v for k, v in sorted(epoch_action_hist.items(), key=lambda x: x[1], reverse=True)
            },
            "reward_normalizer": reward_normalizer.state_dict(),
        }

        eval_out = evaluate_policy_greedy(
            model=policy_model,
            dataset=eval_ds,
            action_space=action_space,
            controller=controller,
            cfg=cfg,
            device=device,
            max_eval_samples=rl_eval_samples,
            temperature=rl_temperature,
        )

        eval_summary = eval_out["summary"]

        merged = {
            **train_summary,
            "eval_accuracy": eval_summary["accuracy"],
            "eval_avg_reward": eval_summary["avg_reward"],
            "eval_avg_prompt_tokens": eval_summary["avg_prompt_tokens"],
            "eval_avg_completion_tokens": eval_summary["avg_completion_tokens"],
            "eval_avg_total_tokens": eval_summary["avg_total_tokens"],
            "eval_action_hist": eval_summary["chosen_action_hist"],
        }
        train_history.append(merged)

        save_json(
            os.path.join(cfg.output_dir, f"rl_epoch_step_records_{epoch}.json"),
            step_records,
        )
        save_json(
            os.path.join(cfg.output_dir, f"rl_eval_records_epoch_{epoch}.json"),
            eval_out["records"],
        )

        torch.save(
            {
                "model_state_dict": policy_model.state_dict(),
                "state_dim": checkpoint["state_dim"],
                "action_dim": checkpoint["action_dim"],
                "base_model_path": base_model_path,
                "epoch": epoch,
                "reward_normalizer": reward_normalizer.state_dict(),
            },
            latest_path,
        )

        if eval_summary["avg_reward"] > best_eval_reward:
            best_eval_reward = eval_summary["avg_reward"]
            torch.save(
                {
                    "model_state_dict": policy_model.state_dict(),
                    "state_dim": checkpoint["state_dim"],
                    "action_dim": checkpoint["action_dim"],
                    "base_model_path": base_model_path,
                    "best_eval_reward": best_eval_reward,
                    "epoch": epoch,
                    "reward_normalizer": reward_normalizer.state_dict(),
                },
                best_eval_path,
            )

        print("\n[EPOCH SUMMARY]")
        print(json.dumps(merged, ensure_ascii=False, indent=2))

    final_eval = evaluate_policy_greedy(
        model=policy_model,
        dataset=eval_ds,
        action_space=action_space,
        controller=controller,
        cfg=cfg,
        device=device,
        max_eval_samples=rl_eval_samples,
        temperature=rl_temperature,
    )

    final_summary = {
        "base_model_path": base_model_path,
        "best_rl_model_path": best_eval_path,
        "final_rl_model_path": latest_path,
        "rl_epochs": rl_epochs,
        "rl_lr": rl_lr,
        "rl_temperature": rl_temperature,
        "rl_entropy_coef": rl_entropy_coef,
        "rl_kl_coef": rl_kl_coef,
        "rl_grad_clip_norm": rl_grad_clip_norm,
        "rl_reward_norm_momentum": rl_reward_norm_momentum,
        "train_split": rl_train_split,
        "train_samples": len(train_ds),
        "eval_split": rl_eval_split,
        "eval_samples": len(eval_ds),
        "final_eval": final_eval["summary"],
    }

    save_json(os.path.join(cfg.output_dir, "rl_train_history.json"), train_history)
    save_json(os.path.join(cfg.output_dir, "rl_final_eval_records.json"), final_eval["records"])
    save_json(os.path.join(cfg.output_dir, "rl_final_summary.json"), final_summary)

    print("\n[FINAL SUMMARY]")
    print(json.dumps(final_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()