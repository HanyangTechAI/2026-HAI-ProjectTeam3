from dataclasses import dataclass
from typing import Any, Dict

from src.controller.action_space import InferenceActionSpace
from src.execution.model_router import get_model_route_config
from src.execution.reasoning_modes import build_prompt
from src.execution.verifier import verify_and_repair_output, extract_final_answer
from src.rewards.heuristic_reward import compute_heuristic_reward


@dataclass
class ControllerExecutionResult:
    action_idx: int
    action_description: Dict[str, Any]
    prompt: str
    raw_text: str
    final_text: str
    extracted_answer: str
    reward_breakdown: Dict[str, float]
    model_name: str
    total_tokens: int
    prompt_tokens: int
    completion_tokens: int
    verification_used: bool
    format_ok: bool


class AdaptiveInferenceController:
    def __init__(self, llm_client, action_space: InferenceActionSpace | None = None):
        self.llm_client = llm_client
        self.action_space = action_space or InferenceActionSpace()

    def execute(self, question: str, gold_answer: str, action_idx: int) -> ControllerExecutionResult:
        action = self.action_space.get_action(action_idx)
        action_desc = self.action_space.describe_action(action_idx)

        model_cfg = get_model_route_config(action.model_route)
        prompt = build_prompt(
            question=question,
            reasoning_budget=action.reasoning_budget,
        )

        response = self.llm_client.generate(
            prompt=prompt,
            model_name=model_cfg.model_name,
        )

        raw_text = response["text"]
        prompt_tokens = int(response.get("prompt_tokens", 0))
        completion_tokens = int(response.get("completion_tokens", 0))
        total_tokens = int(response.get("total_tokens", prompt_tokens + completion_tokens))

        if action.verify:
            verification = verify_and_repair_output(raw_text)
            final_text = verification.repaired_text
            extracted_answer = verification.extracted_answer
            format_ok = verification.format_ok
        else:
            final_text = raw_text
            extracted_answer = extract_final_answer(raw_text)
            format_ok = bool(extracted_answer)

        reward = compute_heuristic_reward(
            pred=extracted_answer,
            gold=gold_answer,
            total_tokens=total_tokens,
            model_cost_multiplier=model_cfg.estimated_cost_multiplier,
            verify_used=action.verify,
            format_ok=format_ok,
            reasoning_budget=action.reasoning_budget,
            model_route=action.model_route,
        )

        return ControllerExecutionResult(
            action_idx=action_idx,
            action_description=action_desc,
            prompt=prompt,
            raw_text=raw_text,
            final_text=final_text,
            extracted_answer=extracted_answer,
            reward_breakdown={
                "total_reward": reward.total_reward,
                "quality_reward": reward.quality_reward,
                "token_cost_penalty": reward.token_cost_penalty,
                "route_cost_penalty": reward.route_cost_penalty,
                "verification_penalty": reward.verification_penalty,
                "format_penalty": reward.format_penalty,
                "reasoning_bonus": reward.reasoning_bonus,
            },
            model_name=model_cfg.model_name,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            verification_used=action.verify,
            format_ok=format_ok,
        )