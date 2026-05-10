from dataclasses import dataclass


@dataclass
class HeuristicRewardResult:
    total_reward: float
    quality_reward: float
    token_cost_penalty: float
    route_cost_penalty: float
    verification_penalty: float
    format_penalty: float
    reasoning_bonus: float


def compute_heuristic_reward(
    pred: str,
    gold: str,
    total_tokens: int,
    model_cost_multiplier: float,
    verify_used: bool,
    format_ok: bool,
    reasoning_budget: str,
    model_route: str,
) -> HeuristicRewardResult:
    quality_reward = 1.0 if pred == gold else 0.0

    token_cost_penalty = 0.0005 * float(total_tokens) * float(model_cost_multiplier)

    if model_route == "large":
        route_cost_penalty = 0.02
    else:
        route_cost_penalty = 0.01

    verification_penalty = 0.01 if verify_used else 0.0
    format_penalty = 0.0 if format_ok else 0.05

    reasoning_bonus = 0.0

    total_reward = (
        quality_reward
        - token_cost_penalty
        - route_cost_penalty
        - verification_penalty
        - format_penalty
        + reasoning_bonus
    )

    return HeuristicRewardResult(
        total_reward=total_reward,
        quality_reward=quality_reward,
        token_cost_penalty=token_cost_penalty,
        route_cost_penalty=route_cost_penalty,
        verification_penalty=verification_penalty,
        format_penalty=format_penalty,
        reasoning_bonus=reasoning_bonus,
    )
