import torch
import torch.nn as nn
import torch.nn.functional as F

from src.prompt_space import PromptAction

class PromptActorCritic(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_instructions: int, num_reasoning: int, num_formats: int, num_self_checks: int, dropout: float = 0.1):
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
        self.reason_head = nn.Linear(hidden_dim, num_reasoning)
        self.format_head = nn.Linear(hidden_dim, num_formats)
        self.self_check_head = nn.Linear(hidden_dim, num_self_checks)

        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        inst_logits = self.inst_head(h)
        reason_logits = self.reason_head(h)
        format_logits = self.format_head(h)
        self_check_logits = self.self_check_head(h)
        value = self.value_head(h).squeeze(-1)
        return inst_logits, reason_logits, format_logits, self_check_logits, value

    def get_dist_and_value(self, logits: torch.Tensor):
        probs = F.softmax(logits, dim=-1)
        probs = probs.clamp(min=1e-8)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        return torch.distributions.Categorical(probs=probs)

    def sample_action(self, x: torch.Tensor):
        inst_logits, reason_logits, format_logits, self_check_logits, value = self.forward(x)
        
        inst_dist = self.get_dist_and_value(inst_logits)
        reason_dist = self.get_dist_and_value(reason_logits)
        format_dist = self.get_dist_and_value(format_logits)
        self_check_dist = self.get_dist_and_value(self_check_logits)
        
        inst_action = inst_dist.sample()
        reason_action = reason_dist.sample()
        format_action = format_dist.sample()
        self_check_action = self_check_dist.sample()
        
        log_prob = inst_dist.log_prob(inst_action) + reason_dist.log_prob(reason_action) + format_dist.log_prob(format_action) + self_check_dist.log_prob(self_check_action)
        
        entropy = inst_dist.entropy() + reason_dist.entropy() + format_dist.entropy() + self_check_dist.entropy()
        
        action = PromptAction(
            instruction_idx=int(inst_action.item()),
            reasoning_idx=int(reason_action.item()),
            format_idx=int(format_action.item()),
            self_check_idx=int(self_check_action.item()),
        )
        
        return action, log_prob, entropy, value

    def evaluate_actions(self, x: torch.Tensor, inst_actions, reason_actions, format_actions, self_check_actions):
        inst_logits, reason_logits, format_logits, self_check_logits, value = self.forward(x)
        
        inst_dist = self.get_dist_and_value(inst_logits)
        reason_dist = self.get_dist_and_value(reason_logits)
        format_dist = self.get_dist_and_value(format_logits)
        self_check_dist = self.get_dist_and_value(self_check_logits)
        
        log_prob = inst_dist.log_prob(inst_actions) + reason_dist.log_prob(reason_actions) + format_dist.log_prob(format_actions) + self_check_dist.log_prob(self_check_actions)
        
        entropy = inst_dist.entropy() + reason_dist.entropy() + format_dist.entropy() + self_check_dist.entropy()
        
        return log_prob, entropy, value

    @torch.no_grad()
    def greedy_action(self, x: torch.Tensor):
        inst_logits, reason_logits, format_logits, self_check_logits, value = self.forward(x)
        
        action = PromptAction(
            instruction_idx=int(inst_logits.argmax(dim=-1).item()),
            reasoning_idx=int(reason_logits.argmax(dim=-1).item()),
            format_idx=int(format_logits.argmax(dim=-1).item()),
            self_check_idx=int(self_check_logits.argmax(dim=-1).item()),
        )
        
        return action, value