from dataclasses import dataclass
from itertools import product


INSTRUCTION_OPTIONS = [
    "Solve the following math word problem carefully.",
    "You are a precise mathematical reasoning assistant. Solve the problem accurately.",
    "Read the problem step by step and compute the final numeric answer.",
]

REASONING_OPTIONS = [
    "",
    "Think step by step before giving the final answer.",
    "Reason carefully, verify intermediate calculations, then provide the final answer.",
]

FORMAT_OPTIONS = [
    "Return only the final answer in the format: FINAL: <number>",
    "At the end, output exactly one line: FINAL: <number>",
]

SELF_CHECK_OPTIONS = [
    "",
    "Double-check your arithmetic before the final answer.",
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

        prompt = "\n".join(part for part in parts if part.strip())
        return prompt