from dataclasses import dataclass

import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from src.data import extract_gold_answer
from src.reward import compute_reward, extract_pred_answer, is_correct


@dataclass
class StepResult:
    reward: float
    correct: bool
    pred: str
    gold: str
    action_idx: int
    total_tokens: int
    predicted_value: float
    advantage: float


class PromptRLTrainer:
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

    def run_single(self, sample, embedding: torch.Tensor):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        x = embedding.to(self.device).unsqueeze(0)

        action, log_prob, entropy, value, _ = self.model.sample_action(x)
        action_idx = int(action.item())

        prompt = self.prompt_space.render_prompt(action_idx, question)
        response = self.llm_client.generate(prompt)

        pred = extract_pred_answer(response.text)
        reward = compute_reward(
            pred=pred,
            gold=gold,
            total_tokens=response.total_tokens,
            reward_correct=self.cfg.reward_correct,
            reward_wrong=self.cfg.reward_wrong,
            token_penalty_coef=self.cfg.token_penalty_coef,
        )

        reward_tensor = torch.tensor([reward], dtype=torch.float32, device=self.device)
        advantage = reward_tensor - value

        policy_loss = -(log_prob * advantage.detach()).mean()
        value_loss = F.mse_loss(value, reward_tensor)
        entropy_loss = -entropy.mean()

        total_loss = (
            policy_loss
            + self.cfg.value_loss_coef * value_loss
            + self.cfg.entropy_coef * entropy_loss
        )

        correct = is_correct(pred, gold)

        return (
            StepResult(
                reward=reward,
                correct=correct,
                pred=pred,
                gold=gold,
                action_idx=action_idx,
                total_tokens=response.total_tokens,
                predicted_value=float(value.item()),
                advantage=float(advantage.item()),
            ),
            total_loss,
            policy_loss,
            value_loss,
            entropy.mean(),
        )

    def train_epoch(self, dataset, embeddings: torch.Tensor):
        self.model.train()

        total_loss_sum = 0.0
        total_policy_loss_sum = 0.0
        total_value_loss_sum = 0.0
        total_entropy_sum = 0.0
        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        total_abs_advantage = 0.0
        action_hist = {}

        pbar = tqdm(range(len(dataset)), desc="train", leave=False)
        for idx in pbar:
            sample = dataset[idx]
            embedding = embeddings[idx]

            result, total_loss, policy_loss, value_loss, entropy = self.run_single(sample, embedding)

            self.optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
            self.optimizer.step()

            total_loss_sum += float(total_loss.item())
            total_policy_loss_sum += float(policy_loss.item())
            total_value_loss_sum += float(value_loss.item())
            total_entropy_sum += float(entropy.item())
            total_reward += result.reward
            total_correct += int(result.correct)
            total_tokens += int(result.total_tokens)
            total_abs_advantage += abs(result.advantage)
            action_hist[result.action_idx] = action_hist.get(result.action_idx, 0) + 1

            seen = idx + 1
            pbar.set_postfix(
                reward=f"{result.reward:.4f}",
                acc=f"{total_correct / seen:.4f}",
                adv=f"{result.advantage:.4f}",
                action=result.action_idx,
            )

        n = len(dataset)
        return {
            "loss": total_loss_sum / n,
            "policy_loss": total_policy_loss_sum / n,
            "value_loss": total_value_loss_sum / n,
            "entropy": total_entropy_sum / n,
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_tokens": total_tokens / n,
            "avg_abs_advantage": total_abs_advantage / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }

    @torch.no_grad()
    def evaluate(self, dataset, embeddings: torch.Tensor):
        self.model.eval()

        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        total_predicted_value = 0.0
        total_abs_value_error = 0.0
        action_hist = {}

        pbar = tqdm(range(len(dataset)), desc="eval", leave=False)
        for idx in pbar:
            sample = dataset[idx]
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            x = embeddings[idx].to(self.device).unsqueeze(0)
            action, value = self.model.greedy_action(x)
            action_idx = int(action.item())
            pred_value = float(value.item())

            action_hist[action_idx] = action_hist.get(action_idx, 0) + 1

            prompt = self.prompt_space.render_prompt(action_idx, question)
            response = self.llm_client.generate(prompt)

            pred = extract_pred_answer(response.text)
            reward = compute_reward(
                pred=pred,
                gold=gold,
                total_tokens=response.total_tokens,
                reward_correct=self.cfg.reward_correct,
                reward_wrong=self.cfg.reward_wrong,
                token_penalty_coef=self.cfg.token_penalty_coef,
            )

            correct = is_correct(pred, gold)

            total_reward += reward
            total_correct += int(correct)
            total_tokens += int(response.total_tokens)
            total_predicted_value += pred_value
            total_abs_value_error += abs(pred_value - reward)

        n = len(dataset)
        return {
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_tokens": total_tokens / n,
            "avg_predicted_value": total_predicted_value / n,
            "avg_abs_value_error": total_abs_value_error / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }