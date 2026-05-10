from dataclasses import dataclass


INSTRUCTION_OPTIONS = [
    "Break the problem into smaller parts internally and compute the exact final numeric answer.",
    "Be extremely careful with arithmetic and compute the exact final numeric answer.",
    "Solve the problem carefully, verify the result internally once, and output the exact final numeric answer.",
    "Identify the quantities and relationships internally, then compute the exact final numeric answer.",
    "Use systematic mathematical reasoning internally and produce the exact final numeric answer.",
]

FIXED_REASONING = "Think carefully internally before answering."
FIXED_FORMAT = "Do not show reasoning. Output only: FINAL: <number>"

FEW_SHOT_SETS = {
    "no_shot": [],
    "one_shot": [
        {
            "question": "A store sold 3 pens for $2 each. How much money did it make?",
            "answer": "6",
        }
    ],
    "two_shot": [
        {
            "question": "A store sold 3 pens for $2 each. How much money did it make?",
            "answer": "6",
        },
        {
            "question": "Tom had 10 apples and gave away 4. How many apples does he have left?",
            "answer": "6",
        },
    ],
    "gsm8k_style_two_shot": [
        {
            "question": "A baker makes 8 trays of cookies with 12 cookies on each tray. He sells 30 cookies. How many cookies does he have left?",
            "answer": "66",
        },
        {
            "question": "A class has 24 students. One fourth of them are absent. How many students are present?",
            "answer": "18",
        },
    ],
}

@dataclass(frozen=True)
class PromptAction:
    instruction_idx: int


class PromptSpace:
    def __init__(self, few_shot_mode: str = "gsm8k_style_two_shot"):
        if few_shot_mode not in FEW_SHOT_SETS:
            raise ValueError(f"Unknown few_shot_mode: {few_shot_mode}")
        
        self.num_instructions = len(INSTRUCTION_OPTIONS)
        self.few_shot_mode = few_shot_mode
        self.examples = FEW_SHOT_SETS[few_shot_mode]

    def render_prompt(self, action: PromptAction, question: str) -> str:
        parts = [
            INSTRUCTION_OPTIONS[action.instruction_idx],
            FIXED_REASONING,
            FIXED_FORMAT,
            "",
        ]

        if self.examples:
            for idx, ex in enumerate(self.examples, start=1):
                parts.extend(
                    [
                        f"Example {idx}",
                        f"Question: {ex['question']}",
                        f"FINAL: {ex['answer']}",
                        "",
                    ]
                )

        parts.append(f"Question: {question}")

        return "\n".join(part for part in parts if part.strip())

    def describe_action(self, action: PromptAction) -> dict:
        return {
            "instruction_idx": action.instruction_idx,
            "instruction": INSTRUCTION_OPTIONS[action.instruction_idx],
            "reasoning": FIXED_REASONING,
            "format": FIXED_FORMAT,
            "few_shot_mode": self.few_shot_mode,
            "num_examples": len(self.examples),
        }
        
    def action_to_string(self, action: PromptAction) -> str:
        return f"inst={action.instruction_idx}|shots={self.few_shot_mode}"