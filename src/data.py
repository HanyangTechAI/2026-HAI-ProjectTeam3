from datasets import load_dataset

from src.demo_samples import load_demo_dataset


def load_gsm8k_subset(dataset_name: str, dataset_config: str, split: str, n_samples: int):
    if dataset_name == "demo" or split == "demo":
        return load_demo_dataset(n_samples=n_samples)

    ds = load_dataset(dataset_name, dataset_config, split=split)
    ds = ds.select(range(min(n_samples, len(ds))))
    return ds


def extract_gold_answer(answer_text: str) -> str:
    marker = "####"
    if marker in answer_text:
        return answer_text.split(marker)[-1].strip()
    return answer_text.strip()
