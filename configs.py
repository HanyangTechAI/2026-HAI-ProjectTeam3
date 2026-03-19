from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset_name: str = "gsm8k"
    dataset_config: str = "main"
    train_split: str = "train"
    test_split: str = "test"

    train_samples: int = 30
    test_samples: int = 20
    exhaustive_samples: int = 8

    model_name: str = "gpt-4.1-mini"
    max_new_tokens: int = 256
    temperature: float = 0.2

    lr: float = 1e-2
    epochs: int = 3
    batch_size: int = 8

    reward_correct: float = 1.0
    reward_wrong: float = 0.0
    token_penalty_coef: float = 0.0005

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