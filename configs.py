from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    test_split: str = "test"

    train_samples: int = 200
    test_samples: int = 100

    model_name: str = "gpt-4.1-mini"
    max_new_tokens: int = 256
    temperature: float = 0.2

    lr: float = 1e-2
    epochs: int = 5
    batch_size: int = 8

    reward_correct: float = 1.0
    reward_wrong: float = 0.0
    token_penalty_coef: float = 0.0005

    seed: int = 42

    api_mode: str = "mock"   # "openai" or "mock"