from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class GRPOTrainOutput:
    loss: float
    mean_advantage: float
    mean_reward: float


def build_state_vector(record: dict):
    feat = record["state_features"]
    handcrafted = [
        feat["normalized_length"],
        feat["normalized_word_count"],
        feat["normalized_digit_count"],
        feat["has_percent"],
        feat["has_money"],
        feat["has_ratio_words"],
        feat["has_multistep_hint"],
    ]
    embedding = record["state_embedding"]
    return handcrafted + embedding


def prepare_grpo_examples(rollout_records: List[dict]) -> List[dict]:
    examples = []

    for row in rollout_records:
        state_vec = build_state_vector(row)
        candidates = row["candidates"]

        rewards = [cand["reward"] for cand in candidates]
        reward_mean = sum(rewards) / len(rewards)

        # 선택적으로 std 정규화
        reward_std = (sum((r - reward_mean) ** 2 for r in rewards) / len(rewards)) ** 0.5
        reward_std = max(reward_std, 1e-6)

        for cand in candidates:
            action_idx = cand["action_idx"]
            reward = cand["reward"]
            advantage = (reward - reward_mean) / reward_std

            examples.append(
                {
                    "state_vec": state_vec,
                    "action_idx": action_idx,
                    "reward": reward,
                    "advantage": advantage,
                }
            )

    return examples


def batchify(examples: List[dict], batch_size: int):
    for i in range(0, len(examples), batch_size):
        yield examples[i : i + batch_size]


def run_grpo_epoch(
    model,
    examples: List[dict],
    optimizer=None,
    device: str = "cpu",
    entropy_coef: float = 0.01,
) -> GRPOTrainOutput:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_advantage = 0.0
    total_reward = 0.0
    total_count = 0

    for batch in batchify(examples, batch_size=32):
        state_x = torch.tensor([ex["state_vec"] for ex in batch], dtype=torch.float32, device=device)
        action_y = torch.tensor([ex["action_idx"] for ex in batch], dtype=torch.long, device=device)
        adv = torch.tensor([ex["advantage"] for ex in batch], dtype=torch.float32, device=device)
        rewards = torch.tensor([ex["reward"] for ex in batch], dtype=torch.float32, device=device)

        logits = model(state_x)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)

        chosen_log_probs = log_probs.gather(1, action_y.unsqueeze(1)).squeeze(1)

        policy_loss = -(adv * chosen_log_probs).mean()

        entropy = -(probs * log_probs).sum(dim=-1).mean()
        loss = policy_loss - entropy_coef * entropy

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        batch_size_actual = len(batch)
        total_loss += float(loss.item()) * batch_size_actual
        total_advantage += float(adv.mean().item()) * batch_size_actual
        total_reward += float(rewards.mean().item()) * batch_size_actual
        total_count += batch_size_actual

    return GRPOTrainOutput(
        loss=total_loss / max(total_count, 1),
        mean_advantage=total_advantage / max(total_count, 1),
        mean_reward=total_reward / max(total_count, 1),
    )