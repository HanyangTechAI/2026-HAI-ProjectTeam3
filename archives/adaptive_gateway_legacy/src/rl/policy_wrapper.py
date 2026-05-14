import torch


class PolicyWrapper:
    def __init__(self, model):
        self.model = model

    def get_action_probs(self, state_features, state_embedding):
        logits = self.model(
            state_features,
            state_embedding
        )
        probs = torch.softmax(logits, dim=-1)
        return probs

    def sample_action(self, state_features, state_embedding):
        probs = self.get_action_probs(state_features, state_embedding)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob