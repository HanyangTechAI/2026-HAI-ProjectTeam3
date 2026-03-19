from dataclasses import dataclass

import torch
import torch.optim as optim
from tqdm import tqdm

from src.data import extract_gold_answer, simple_question_features
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

    def run_single(self, sample):
        question = sample["question"]
        gold = extract_gold_answer(sample["answer"])

        features = torch.tensor(
            simple_question_features(question),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        action, log_prob, entropy, _ = self.policy.sample_action(features)
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

        correct = is_correct(pred, gold)

        return (
            StepResult(
                reward=reward,
                correct=correct,
                pred=pred,
                gold=gold,
                action_idx=action_idx,
                total_tokens=response.total_tokens,
            ),
            log_prob,
            entropy,
        )

    def train_epoch(self, dataset):
        self.policy.train()
        total_loss = 0.0
        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        action_hist = {}

        pbar = tqdm(dataset, desc="train", leave=False)
        for idx, sample in enumerate(pbar, start=1):
            result, log_prob, entropy = self.run_single(sample)

            loss = -(log_prob * result.reward) - 0.001 * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += float(loss.item())
            total_reward += result.reward
            total_correct += int(result.correct)
            total_tokens += int(result.total_tokens)
            action_hist[result.action_idx] = action_hist.get(result.action_idx, 0) + 1

            pbar.set_postfix(
                reward=f"{result.reward:.4f}",
                acc=f"{total_correct / idx:.4f}",
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
    def evaluate(self, dataset):
        self.policy.eval()
        total_reward = 0.0
        total_correct = 0
        total_tokens = 0
        action_hist = {}

        pbar = tqdm(dataset, desc="eval", leave=False)
        for sample in pbar:
            question = sample["question"]
            gold = extract_gold_answer(sample["answer"])

            features = torch.tensor(
                simple_question_features(question),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            logits = self.policy(features)
            action_idx = int(torch.argmax(logits, dim=-1).item())

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

        n = len(dataset)
        return {
            "reward": total_reward / n,
            "accuracy": total_correct / n,
            "avg_tokens": total_tokens / n,
            "action_hist": dict(sorted(action_hist.items(), key=lambda x: x[1], reverse=True)),
        }