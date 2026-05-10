from dataclasses import dataclass


@dataclass
class TrainConfig:
    # common
    seed: int = 42
    output_dir: str = "outputs"

    # dataset
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    test_split: str = "test"
    train_samples: int = 50
    test_samples: int = 30

<<<<<<< Updated upstream
    train_samples: int = 40
    test_samples: int = 15
    exhaustive_samples: int = 10

    model_name: str = "gpt-4.1-mini"
    max_new_tokens: int = 64
    temperature: float = 0.2

    lr: float = 1e-4
    epochs: int = 10
    batch_size: int = 8

    reward_correct: float = 1.0
    reward_wrong: float = 0.0

    prompt_token_penalty_coef: float = 0.0002
    completion_token_penalty_coef: float = 0.0010

    entropy_coef: float = 0.05
    value_loss_coef: float = 0.5
    grad_clip_norm: float = 0.5
    
    ppo_clip_eps: float = 0.2
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
=======
    # oracle rollout range
    oracle_split: str = "train"
    oracle_start_idx: int = 0
    oracle_num_samples: int = 50
>>>>>>> Stashed changes

    # embedding / state encoding
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # api / llm
    # Use "mock" for offline demos and "openai" for real model calls.
    api_mode: str = "openai"

    # demo
    demo_embedding_model_name: str = "hashing:384"

    # efficiency pair
    efficiency_reward_margin: float = 0.01
    efficiency_min_cost_gap: int = 1

    # balanced combined
    balanced_efficiency_ratio: float = 0.5

    # RL config
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
