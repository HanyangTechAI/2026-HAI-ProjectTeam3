import re


def normalize_number_string(text: str) -> str:
    return text.strip().replace(",", "")


def extract_pred_answer(text: str) -> str:
    match = re.search(r"FINAL:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_number_string(match.group(1))

    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return normalize_number_string(numbers[-1])

    return ""


def is_correct(pred: str, gold: str) -> bool:
    return normalize_number_string(pred) == normalize_number_string(gold)


def compute_reward(
    pred: str,
    gold: str,
    prompt_tokens: int,
    completion_tokens: int,
    reward_correct: float,
    reward_wrong: float,
    prompt_token_penalty_coef: float,
    completion_token_penalty_coef: float,
) -> float:
    base_reward = reward_correct if is_correct(pred, gold) else reward_wrong

    prompt_penalty = prompt_token_penalty_coef * float(prompt_tokens)
    completion_penalty = completion_token_penalty_coef * float(completion_tokens)

    return base_reward - prompt_penalty - completion_penalty