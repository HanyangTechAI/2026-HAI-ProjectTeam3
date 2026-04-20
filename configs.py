from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    test_split: str = "test"

    train_samples: int = 60
    test_samples: int = 15
    exhaustive_samples: int = 10

    model_name: str = "gpt-4.1-mini"
    max_new_tokens: int = 32
    temperature: float = 0.2

    lr: float = 5e-5
    epochs: int = 10
    batch_size: int = 8

    reward_correct: float = 1.0
    reward_wrong: float = 0.0

    prompt_token_penalty_coef: float = 0.0002
    completion_token_penalty_coef: float = 0.0002

    entropy_coef: float = 0.02
    value_loss_coef: float = 0.5
    grad_clip_norm: float = 0.5
    
    ppo_clip_eps: float = 0.1
    ppo_update_epochs: int = 4
    normalize_advantage: bool = True

    seed: int = 42
    api_mode: str = "openai"   # "openai" or "mock"

    fixed_action_indices: tuple[int, ...] = (0, 10, 20, 35)

    output_dir: str = "outputs"
    train_log_json: str = "train_history.json"
    train_log_csv: str = "train_history.csv"
    leaderboard_json: str = "leaderboard.json"
    leaderboard_csv: str = "leaderboard.csv"
    exhaustive_json: str = "exhaustive.json"
    random_json: str = "random_baseline.json"
    rl_policy_json: str = "rl_policy.json"

    train_curve_png: str = "train_curves.png"
    action_hist_png: str = "action_hist.png"
    baseline_bar_png: str = "baseline_accuracy.png"
    exhaustive_action_png: str = "exhaustive_action_scores.png"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    normalize_embeddings: bool = True
    embedding_batch_size: int = 64
    train_embedding_cache: str = "train_embeddings.pt"
    test_embedding_cache: str = "test_embeddings.pt"
    exhaustive_embedding_cache: str = "exhaustive_embeddings.pt"

    policy_hidden_dim: int = 256
    policy_dropout: float = 0.1