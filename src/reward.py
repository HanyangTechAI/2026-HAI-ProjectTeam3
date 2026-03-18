import re


def normalize_number_string(text: str) -> str:
    text = text.strip()
    text = text.replace(",", "")
    return text


def extract_pred_answer(text: str) -> str:
    """
    우선순위:
    1. FINAL: <answer>
    2. 마지막 숫자
    """
    match = re.search(r"FINAL:\s*([-+]?\d*\.?\d+)", text, flags=re.IGNORECASE)
    if match:
        return normalize_number_string(match.group(1))

    numbers = re.findall(r"[-+]?\d*\.?\d+", text.replace(",", ""))
    if numbers:
        return normalize_number_string(numbers[-1])

    return ""


def is_correct(pred: str, gold: str) -> bool:
    return normalize_number_string(pred) == normalize_number_string(gold)


def compute_reward(pred: str, gold: str, total_tokens: int, reward_correct: float, reward_wrong: float, token_penalty_coef: float) -> float:
    base = reward_correct if is_correct(pred, gold) else reward_wrong
    penalty = token_penalty_coef * float(total_tokens)
    return base - penalty