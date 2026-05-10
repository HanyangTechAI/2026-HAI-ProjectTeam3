from src.rewards.heuristic_reward import compute_heuristic_reward


class RewardModel:
    def compute(self, result, gold):
        reward = compute_heuristic_reward(
            pred=result["pred"],
            gold=gold,
            total_tokens=result["total_tokens"],
            model_cost_multiplier=1.0,
            verify_used=result["verification_used"],
            format_ok=result["format_ok"],
            reasoning_budget=result["action"]["reasoning_budget"],
            model_route=result["action"]["model_route"],
        )
        return reward.total_reward