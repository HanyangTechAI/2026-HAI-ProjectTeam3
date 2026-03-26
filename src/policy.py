import torch
import torch.nn as nn
import torch.nn.functional as F


class PromptActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int, dropout: float = 0.1):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.policy_head = nn.Linear(hidden_dim, n_actions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value

    def sample_action(self, x: torch.Tensor):
        logits, value = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value, probs

    @torch.no_grad()
    def greedy_action(self, x: torch.Tensor):
        logits, value = self.forward(x)
        action = torch.argmax(logits, dim=-1)
        return action, value