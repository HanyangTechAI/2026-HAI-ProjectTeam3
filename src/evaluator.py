from dataclasses import dataclass, field
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
    total_tokens: int
    action_idx: int
    raw_text: str

    rollout_rewards: list[float] = field(default_factory=list)
    rollout_corrects: list[bool] = field(default_factory=list)
    rollout_tokens: list[int] = field(default_factory=list)
    rollout_preds: list[str] = field(default_factory=list)
    rollout_raw_texts: list[str] = field(default_factory=list)


def evaluate_single_action_on_sample(sample: Any, action_idx: int, prompt_space, llm_client, cfg) -> EvalRecord:
    question = sample["question"]
    gold = extract_gold_answer(sample["answer"])

    prompt = prompt_space.render_prompt(action_idx, question)

    n_rollouts = getattr(cfg, "n_reward_rollouts", 1)

    rewards = []
    corrects = []
    total_tokens_list = []
    preds = []
    raw_texts = []

    for _ in range(n_rollouts):
        response = llm_client.generate(prompt)

        pred = extract_pred_answer(response.text)
        correct = is_correct(pred, gold)
        reward = compute_reward(
            pred=pred,
            gold=gold,
            total_tokens=response.total_tokens,
            reward_correct=cfg.reward_correct,
            reward_wrong=cfg.reward_wrong,
            token_penalty_coef=cfg.token_penalty_coef,
        )

        rewards.append(reward)
        corrects.append(correct)
        total_tokens_list.append(response.total_tokens)
        preds.append(pred)
        raw_texts.append(response.text)

    avg_reward = sum(rewards) / len(rewards)
    avg_correct = (sum(int(c) for c in corrects) / len(corrects)) >= 0.5
    avg_tokens = int(round(sum(total_tokens_list) / len(total_tokens_list)))

    return EvalRecord(
        question=question,
        gold=gold,
        pred=preds[-1],
        correct=avg_correct,
        reward=avg_reward,
        total_tokens=avg_tokens,
        action_idx=action_idx,
        raw_text=raw_texts[-1],
        rollout_rewards=rewards,
        rollout_corrects=corrects,
        rollout_tokens=total_tokens_list,
        rollout_preds=preds,
        rollout_raw_texts=raw_texts,
    )