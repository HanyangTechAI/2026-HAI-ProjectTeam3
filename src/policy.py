import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prompt_space import PromptAction

class PromptActorCritic(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        num_instructions: int, 
        dropout: float = 0.1):
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

        self.inst_head = nn.Linear(hidden_dim, num_instructions)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        inst_logits = self.inst_head(h)
        value = self.value_head(h).squeeze(-1)
        return inst_logits, value

    def get_dist_and_value(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=probs)

    def sample_action(self, x: torch.Tensor):
        inst_logits, value = self.forward(x)
        inst_dist = self.get_dist_and_value(inst_logits)
        inst_action = inst_dist.sample()
        
        log_prob = inst_dist.log_prob(inst_action)
        entropy = inst_dist.entropy()
        
        action = PromptAction(
            instruction_idx=int(inst_action.item()),
        )
        
        return action, log_prob, entropy, value

    def evaluate_actions(self, x: torch.Tensor, inst_actions: torch.Tensor):
        inst_logits, value = self.forward(x)
        inst_dist = self.get_dist_and_value(inst_logits)
        
        log_prob = inst_dist.log_prob(inst_actions)
        entropy = inst_dist.entropy()
        
        return log_prob, entropy, value

    @torch.no_grad()
    def greedy_action(self, x: torch.Tensor):
        inst_logits, value = self.forward(x)
        
        action = PromptAction(
            instruction_idx=int(inst_logits.argmax(dim=-1).item()),
        )
        
        return action, value