from dataclasses import dataclass


@dataclass
class TrainConfig:
    # common
    seed: int = 42
    output_dir: str = "outputs"
    service_log_path: str = "outputs/service_requests.jsonl"

    # legacy math benchmark dataset
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    test_split: str = "test"
    train_samples: int = 50
    test_samples: int = 30

    # oracle rollout range for preference datasets/checkpoints
    oracle_split: str = "train"
    oracle_start_idx: int = 0
    oracle_num_samples: int = 50

    # embedding / state encoding
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    demo_embedding_model_name: str = "hashing:384"

    # api / llm. Use "mock" for offline demos and "openai" for real model calls.
    api_mode: str = "openai"

    # preference pair construction
    efficiency_reward_margin: float = 0.01
    efficiency_min_cost_gap: int = 1
    balanced_efficiency_ratio: float = 0.5

    # RL fine-tuning
    rl_train_split: str = "train"
    rl_train_samples: int = 30
    rl_eval_split: str = "test"
    rl_eval_samples: int = 30
    rl_epochs: int = 3
    rl_lr: float = 1e-5
    rl_temperature: float = 1.0
    rl_entropy_coef: float = 0.01
    rl_kl_coef: float = 0.05
    rl_grad_clip_norm: float = 1.0
    rl_reward_norm_momentum: float = 0.95
