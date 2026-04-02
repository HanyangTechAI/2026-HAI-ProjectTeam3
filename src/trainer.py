from dataclasses import dataclass

import torch
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


class PromptRLTrainer:
    def __init__(
        self,
        policy,
        prompt_space,
        llm_client,
        train_config,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.prompt_space = prompt_space
        self.llm_client = llm_client
        self.cfg = train_config
        self.device = device
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.cfg.lr)

    def run_single(self, sample, embedding: torch.Tensor):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        x = embedding.to(self.device).unsqueeze(0)

        action, log_prob, entropy, _ = self.policy.sample_action(x)
        action_idx = int(action.item())

        prompt = self.prompt_space.render_prompt(action_idx, question)

        n_rollouts = getattr(self.cfg, "n_reward_rollouts", 1)

        rewards = []
        corrects = []
        total_tokens_list = []
        preds = []

        for _ in range(n_rollouts):
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

            rewards.append(reward)
            corrects.append(correct)
            total_tokens_list.append(response.total_tokens)
            preds.append(pred)

        avg_reward = sum(rewards) / len(rewards)
        avg_correct = (sum(int(c) for c in corrects) / len(corrects)) >= 0.5
        avg_tokens = int(round(sum(total_tokens_list) / len(total_tokens_list)))

        return (
            StepResult(
                reward=avg_reward,
                correct=avg_correct,
                pred=preds[-1],
                gold=gold,
                action_idx=action_idx,
                total_tokens=avg_tokens,
            ),
            log_prob,
            entropy,
        )

    def train_epoch(self, dataset, embeddings: torch.Tensor):
        self.policy.train()
        total_loss = 0.0
        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        action_hist = {}

        pbar = tqdm(range(len(dataset)), desc="train", leave=False)
        for idx in pbar:
            sample = dataset[idx]
            embedding = embeddings[idx]

            result, log_prob, entropy = self.run_single(sample, embedding)

            loss = -(log_prob * result.reward) - self.cfg.entropy_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_reward += result.reward
            total_correct += int(result.correct)
            total_tokens += int(result.total_tokens)
            action_hist[result.action_idx] = action_hist.get(result.action_idx, 0) + 1

            seen = idx + 1
            pbar.set_postfix(
                reward=f"{result.reward:.4f}",
                acc=f"{total_correct / seen:.4f}",
                action=result.action_idx,
            )

        n = len(dataset)
        return {
            "loss": total_loss / n,
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_tokens": total_tokens / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }

    @torch.no_grad()
    def evaluate(self, dataset, embeddings: torch.Tensor):
        self.policy.eval()
        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        action_hist = {}

        pbar = tqdm(range(len(dataset)), desc="eval", leave=False)
        for idx in pbar:
            sample = dataset[idx]
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            x = embeddings[idx].to(self.device).unsqueeze(0)
            logits = self.policy(x)
            action_idx = int(torch.argmax(logits, dim=-1).item())
            action_hist[action_idx] = action_hist.get(action_idx, 0) + 1

            prompt = self.prompt_space.render_prompt(action_idx, question)
            n_rollouts = getattr(self.cfg, "n_reward_rollouts", 1)

            rewards = []
            corrects = []
            tokens_list = []

            for _ in range(n_rollouts):
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

                rewards.append(reward)
                corrects.append(correct)
                tokens_list.append(response.total_tokens)

            total_reward += sum(rewards) / len(rewards)
            total_correct += int((sum(int(c) for c in corrects) / len(corrects)) >= 0.5)
            total_tokens += int(round(sum(tokens_list) / len(tokens_list)))

        n = len(dataset)
        return {
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_tokens": total_tokens / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }