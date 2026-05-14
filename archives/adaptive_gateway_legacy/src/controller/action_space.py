from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass(frozen=True)
class InferenceAction:
    action_idx: int
    reasoning_budget: str   # "none" | "short" | "long"
    model_route: str        # "small" | "large"
    verify: bool


class InferenceActionSpace:
    def __init__(self):
        self.actions: List[InferenceAction] = [
            InferenceAction(0, "none",  "small", False),
            InferenceAction(1, "none",  "small", True),
            InferenceAction(2, "short", "small", False),
            InferenceAction(3, "short", "small", True),
            InferenceAction(4, "short", "large", False),
            InferenceAction(5, "short", "large", True),
            InferenceAction(6, "long",  "small", False),
            InferenceAction(7, "long",  "large", False),
        ]

    def __len__(self) -> int:
        return len(self.actions)

    def get_action(self, action_idx: int) -> InferenceAction:
        if action_idx < 0 or action_idx >= len(self.actions):
            raise IndexError(f"Invalid action_idx: {action_idx}")
        return self.actions[action_idx]

    def describe_action(self, action_idx: int) -> Dict[str, Any]:
        action = self.get_action(action_idx)
        return {
            "action_idx": action.action_idx,
            "reasoning_budget": action.reasoning_budget,
            "model_route": action.model_route,
            "verify": action.verify,
        }

    def all_actions(self) -> List[InferenceAction]:
        return self.actions[:]