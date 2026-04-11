from dataclasses import dataclass
from itertools import product


INSTRUCTION_OPTIONS = [
    "Solve the math word problem carefully.",
    "Read the problem and compute the correct final numeric answer.",
    "You are a precise math assistant. Solve the problem accurately.",
]

REASONING_OPTIONS = [
    "",
    "Think carefully internally before answering.",
]

FORMAT_OPTIONS = [
    "Output exactly one line: FINAL: <number>",
    "Do not show reasoning. Output only: FINAL: <number>",
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
        self.num_instructions = len(INSTRUCTION_OPTIONS)
        self.num_reasoning = len(REASONING_OPTIONS)
        self.num_formats = len(FORMAT_OPTIONS)
        self.num_self_checks = len(SELF_CHECK_OPTIONS)

    def render_prompt(self, action: PromptAction, question: str) -> str:
        parts = [
            INSTRUCTION_OPTIONS[action.instruction_idx],
            REASONING_OPTIONS[action.reasoning_idx],
            SELF_CHECK_OPTIONS[action.self_check_idx],
            FORMAT_OPTIONS[action.format_idx],
            "",
            f"Question: {question}",
        ]
        return "\n".join(part for part in parts if part.strip())

    def describe_action(self, action: PromptAction) -> dict:
        return {
            "instruction_idx": action.instruction_idx,
            "reasoning_idx": action.reasoning_idx,
            "format_idx": action.format_idx,
            "self_check_idx": action.self_check_idx,
            "instruction": INSTRUCTION_OPTIONS[action.instruction_idx],
            "reasoning": REASONING_OPTIONS[action.reasoning_idx],
            "format": FORMAT_OPTIONS[action.format_idx],
            "self_check": SELF_CHECK_OPTIONS[action.self_check_idx],
        }