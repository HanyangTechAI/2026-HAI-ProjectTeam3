from dataclasses import dataclass
from itertools import product


INSTRUCTION_OPTIONS = [
    "Solve the following math word problem carefully.",
    "You are a precise mathematical reasoning assistant. Solve the problem accurately.",
    "Read the problem carefully and compute the final numeric answer.",
]

REASONING_OPTIONS = [
    "",
    "Think carefully internally before answering.",
    "Reason through the problem internally, verify the calculation internally, and then output only the final answer.",
]

FORMAT_OPTIONS = [
    "Do not show any reasoning steps. Output exactly one line in this format: FINAL: <number>",
    "Your entire response must be exactly one line and nothing else: FINAL: <number>",
]

SELF_CHECK_OPTIONS = [
    "",
    "Double-check your arithmetic internally before outputting the final answer.",
]


@dataclass(frozen=True)
class PromptAction:
    instruction_idx: int
    reasoning_idx: int
    format_idx: int
    self_check_idx: int


class PromptSpace:
    def __init__(self):
        self.actions = [
            PromptAction(i, r, f, s)
            for i, r, f, s in product(
                range(len(INSTRUCTION_OPTIONS)),
                range(len(REASONING_OPTIONS)),
                range(len(FORMAT_OPTIONS)),
                range(len(SELF_CHECK_OPTIONS)),
            )
        ]

    def __len__(self):
        return len(self.actions)

    def get_action(self, idx: int) -> PromptAction:
        return self.actions[idx]

    def describe_action(self, idx: int) -> dict:
        action = self.get_action(idx)
        return {
            "action_idx": idx,
            "instruction": INSTRUCTION_OPTIONS[action.instruction_idx],
            "reasoning": REASONING_OPTIONS[action.reasoning_idx],
            "format": FORMAT_OPTIONS[action.format_idx],
            "self_check": SELF_CHECK_OPTIONS[action.self_check_idx],
        }

    def render_prompt(self, action_idx: int, question: str) -> str:
        action = self.get_action(action_idx)
        parts = [
            INSTRUCTION_OPTIONS[action.instruction_idx],
            REASONING_OPTIONS[action.reasoning_idx],
            SELF_CHECK_OPTIONS[action.self_check_idx],
            FORMAT_OPTIONS[action.format_idx],
            "",
            f"Question: {question}",
        ]
        return "\n".join(part for part in parts if part.strip())