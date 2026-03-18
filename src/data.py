from datasets import load_dataset


def load_gsm8k_subset(dataset_name: str, dataset_config: str, split: str, n_samples: int):
    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def extract_gold_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker)[-1].strip()
    return answer_text.strip()


def simple_question_features(question: str) -> list[float]:
    length = len(question)
    digit_count = sum(ch.isdigit() for ch in question)
    has_percent = 1.0 if "%" in question else 0.0
    has_money = 1.0 if "$" in question else 0.0
    has_more_than_one_sentence = 1.0 if question.count(".") + question.count("?") > 1 else 0.0

    return [
        float(length) / 300.0,
        float(digit_count) / 20.0,
        has_percent,
        has_money,
        has_more_than_one_sentence,
    ]