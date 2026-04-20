from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.data import extract_gold_answer
from src.reward import compute_reward, extract_pred_answer, is_correct


@dataclass
class RolloutItem:
    embedding: torch.Tensor
    inst_action: int
    old_log_prob: float
    reward: float
    correct: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    value: float
    action_str: str


class PromptPPOTrainer:
    def __init__(
        self,
        model,
        prompt_space,
        llm_client,
        train_config,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.prompt_space = prompt_space
        self.llm_client = llm_client
        self.cfg = train_config
        self.device = device
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.lr)

    def collect_rollout(self, dataset, embeddings: torch.Tensor):
        self.model.eval()

        rollout = []
        action_hist = {}
        total_reward = 0.0
        total_correct = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0

        pbar = tqdm(range(len(dataset)), desc="collect", leave=False)
        for idx in pbar:
            sample = dataset[idx]
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            x = embeddings[idx].to(self.device).unsqueeze(0)

            with torch.no_grad():
                action, log_prob, _, value = self.model.sample_action(x)

            prompt = self.prompt_space.render_prompt(action, question)
            response = self.llm_client.generate(prompt)

            pred = extract_pred_answer(response.text)
            reward = compute_reward(
                pred=pred,
                gold=gold,
                completion_tokens=response.completion_tokens,
                reward_correct=self.cfg.reward_correct,
                reward_wrong=self.cfg.reward_wrong,
                completion_token_penalty_coef=self.cfg.completion_token_penalty_coef,
            )
            correct = is_correct(pred, gold)

            action_str = self.prompt_space.action_to_string(action)

            rollout.append(
                RolloutItem(
                    embedding=embeddings[idx].detach().cpu(),
                    inst_action=action.instruction_idx,
                    old_log_prob=log_prob.item(),
                    reward=float(reward),
                    correct=bool(correct),
                    prompt_tokens=int(response.prompt_tokens),
                    completion_tokens=int(response.completion_tokens),
                    total_tokens=int(response.total_tokens),
                    value=float(value.item()),
                    action_str=action_str,
                )
            )

            total_reward += float(reward)
            total_correct += int(correct)
            total_prompt_tokens += int(response.prompt_tokens)
            total_completion_tokens += int(response.completion_tokens)
            total_tokens += int(response.total_tokens)
            action_hist[action_str] = action_hist.get(action_str, 0) + 1

        n = len(dataset)
        stats = {
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_prompt_tokens": total_prompt_tokens / n,
            "avg_completion_tokens": total_completion_tokens / n,
            "avg_tokens": total_tokens / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }
        return rollout, stats

    def _prepare_tensors(self, rollout):
        x = torch.stack([item.embedding for item in rollout], dim=0).to(self.device)
        inst_actions = torch.tensor([item.inst_action for item in rollout], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([item.old_log_prob for item in rollout], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([item.reward for item in rollout], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([item.value for item in rollout], dtype=torch.float32, device=self.device)

        advantages = rewards - old_values
        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        returns = rewards
        return (
            x,
            inst_actions,
            old_log_probs,
            advantages,
            returns,
        )

    def train_epoch(self, dataset, embeddings: torch.Tensor):
        rollout, collect_stats = self.collect_rollout(dataset, embeddings)

        (
            x,
            inst_actions,
            old_log_probs,
            advantages,
            returns,
        ) = self._prepare_tensors(rollout)

        self.model.train()

        total_loss_sum = 0.0
        total_policy_loss_sum = 0.0
        total_value_loss_sum = 0.0
        total_entropy_sum = 0.0

        for _ in range(self.cfg.ppo_update_epochs):
            new_log_probs, entropy, values = self.model.evaluate_actions(
                x,
                inst_actions,
            )

            ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - self.cfg.ppo_clip_eps,
                1.0 + self.cfg.ppo_clip_eps,
            )

            surrogate1 = ratio * advantages
            surrogate2 = clipped_ratio * advantages
            policy_loss = -torch.min(surrogate1, surrogate2).mean()

            value_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy.mean()

            total_loss = (
                policy_loss
                + self.cfg.value_loss_coef * value_loss
                + self.cfg.entropy_coef * entropy_loss
            )

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.cfg.grad_clip_norm,
            )
            self.optimizer.step()

            total_loss_sum += float(total_loss.item())
            total_policy_loss_sum += float(policy_loss.item())
            total_value_loss_sum += float(value_loss.item())
            total_entropy_sum += float(entropy.mean().item())

        denom = float(self.cfg.ppo_update_epochs)

        return {
            "loss": total_loss_sum / denom,
            "policy_loss": total_policy_loss_sum / denom,
            "value_loss": total_value_loss_sum / denom,
            "entropy": total_entropy_sum / denom,
            "reward": collect_stats["reward"],
            "accuracy": collect_stats["accuracy"],
            "avg_prompt_tokens": collect_stats["avg_prompt_tokens"],
            "avg_completion_tokens": collect_stats["avg_completion_tokens"],
            "avg_tokens": collect_stats["avg_tokens"],
            "avg_abs_advantage": float(torch.abs(advantages).mean().item()),
            "action_hist": collect_stats["action_hist"],
        }

    @torch.no_grad()
    def evaluate(self, dataset, embeddings: torch.Tensor):
        self.model.eval()

        total_reward = 0.0
        total_correct = 0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        total_predicted_value = 0.0
        total_abs_value_error = 0.0
        action_hist = {}

        for idx in range(len(dataset)):
            sample = dataset[idx]
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            x = embeddings[idx].to(self.device).unsqueeze(0)
            action, value = self.model.greedy_action(x)

            prompt = self.prompt_space.render_prompt(action, question)
            response = self.llm_client.generate(prompt)

            pred = extract_pred_answer(response.text)
            reward = compute_reward(
                pred=pred,
                gold=gold,
                completion_tokens=response.completion_tokens,
                reward_correct=self.cfg.reward_correct,
                reward_wrong=self.cfg.reward_wrong,
                completion_token_penalty_coef=self.cfg.completion_token_penalty_coef,
            )

            correct = is_correct(pred, gold)

            total_reward += float(reward)
            total_correct += int(correct)
            total_prompt_tokens += int(response.prompt_tokens)
            total_completion_tokens += int(response.completion_tokens)
            total_tokens += int(response.total_tokens)
            total_predicted_value += float(value.item())
            total_abs_value_error += abs(float(value.item()) - float(reward))

            action_str = self.prompt_space.action_to_string(action)
            action_hist[action_str] = action_hist.get(action_str, 0) + 1

        n = len(dataset)
        return {
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_prompt_tokens": total_prompt_tokens / n,
            "avg_completion_tokens": total_completion_tokens / n,
            "avg_tokens": total_tokens / n,
            "avg_predicted_value": total_predicted_value / n,
            "avg_abs_value_error": total_abs_value_error / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }