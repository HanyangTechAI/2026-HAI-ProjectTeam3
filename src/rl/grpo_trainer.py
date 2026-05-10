import torch
from torch.optim import Adam


class GRPOTrainer:
    def __init__(self, policy, reward_model, controller, lr=1e-5):
        self.policy = policy
        self.reward_model = reward_model
        self.controller = controller
        self.optimizer = Adam(policy.model.parameters(), lr=lr)

    def train_step(self, sample):
        question = sample["question"]
        gold = sample["gold"]
        state_features = sample["state_features"]
        state_embedding = sample["state_embedding"]

        # 1. action sampling
        action, log_prob = self.policy.sample_action(
            state_features, state_embedding
        )

        # 2. rollout
        result = self.controller.execute(
            question, gold, action
        )

        # 3. reward
        reward = self.reward_model.compute(result, gold)

        # 4. policy gradient
        loss = -log_prob * reward

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "reward": reward,
            "action": action,
            "loss": loss.item()
        }