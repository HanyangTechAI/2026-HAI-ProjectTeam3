from dataclasses import dataclass
from typing import Any

from src.data import extract_gold_answer
from src.reward import compute_reward, extract_pred_answer, is_correct


@dataclass
class EvalRecord:
    question: str
    gold: str
    pred: str
    correct: bool
    reward: float
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    action_idx: int
    raw_text: str


def evaluate_single_action_on_sample(
    sample: Any,
    action_idx: int,
    prompt_space,
    llm_client,
    cfg,
) -> EvalRecord:
    question = sample["question"]
    gold = extract_gold_answer(sample["answer"])

    prompt = prompt_space.render_prompt(action_idx, question)
    response = llm_client.generate(prompt)

    pred = extract_pred_answer(response.text)
    correct = is_correct(pred, gold)

    reward = compute_reward(
        pred=pred,
        gold=gold,
        completion_tokens=response.completion_tokens,
        reward_correct=cfg.reward_correct,
        reward_wrong=cfg.reward_wrong,
        completion_token_penalty_coef=cfg.completion_token_penalty_coef,
    )

    return EvalRecord(
        question=question,
        gold=gold,
        pred=pred,
        correct=correct,
        reward=reward,
        prompt_tokens=int(response.prompt_tokens),
        completion_tokens=int(response.completion_tokens),
        total_tokens=int(response.total_tokens),
        action_idx=action_idx,
        raw_text=response.text,
    )