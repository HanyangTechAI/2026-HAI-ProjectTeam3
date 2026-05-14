import torch
import torch.nn as nn
import torch.nn.functional as F


class ActionPreferenceNet(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()

        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),   # net.0
            nn.LayerNorm(hidden_dim),           # net.1
            nn.ReLU(),                          # net.2
            nn.Dropout(dropout),                # net.3
            nn.Linear(hidden_dim, hidden_dim),  # net.4
            nn.ReLU(),                          # net.5
            nn.Dropout(dropout),                # net.6
            nn.Linear(hidden_dim, 1),           # net.7
        )

    def forward(self, state_x: torch.Tensor, action_x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_x: [B, state_dim]
            action_x: [B, action_dim]

        Returns:
            score: [B, 1]
        """
        x = torch.cat([state_x, action_x], dim=-1)
        return self.net(x)

    def score_difference(
        self,
        state_x: torch.Tensor,
        preferred_action_x: torch.Tensor,
        rejected_action_x: torch.Tensor,
    ) -> torch.Tensor:
        preferred_score = self.forward(state_x, preferred_action_x)
        rejected_score = self.forward(state_x, rejected_action_x)
        return preferred_score - rejected_score

    def preference_loss(
        self,
        state_x: torch.Tensor,
        preferred_action_x: torch.Tensor,
        rejected_action_x: torch.Tensor,
    ) -> torch.Tensor:
        diff = self.score_difference(
            state_x=state_x,
            preferred_action_x=preferred_action_x,
            rejected_action_x=rejected_action_x,
        )
        return -F.logsigmoid(diff).mean()