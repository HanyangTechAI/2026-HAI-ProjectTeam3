import os

import matplotlib.pyplot as plt
import pandas as pd

from archives.utils import load_json


def plot_train_curves(train_history_path: str, output_path: str) -> None:
    history = load_json(train_history_path)
    if not history:
        return

    df = pd.DataFrame(history)
    if df.empty:
        return

    plt.figure(figsize=(10, 6))
    plt.plot(df["epoch"], df["train_accuracy"], label="train_accuracy")
    plt.plot(df["epoch"], df["eval_accuracy"], label="eval_accuracy")
    plt.plot(df["epoch"], df["train_reward"], label="train_reward")
    plt.plot(df["epoch"], df["eval_reward"], label="eval_reward")
    plt.xlabel("epoch")
    plt.ylabel("value")
    plt.title("Training Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_action_hist(action_hist: dict, output_path: str, title: str = "Action Histogram") -> None:
    if not action_hist:
        return

    items = sorted(action_hist.items(), key=lambda x: int(x[0]) if str(x[0]).isdigit() else x[0])
    x = [str(k) for k, _ in items]
    y = [v for _, v in items]

    plt.figure(figsize=(12, 6))
    plt.bar(x, y)
    plt.xlabel("action_idx")
    plt.ylabel("count")
    plt.title(title)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_baseline_accuracy(leaderboard_path: str, output_path: str) -> None:
    leaderboard = load_json(leaderboard_path)
    if not leaderboard:
        return

    df = pd.DataFrame(leaderboard)
    if df.empty or "name" not in df.columns or "accuracy" not in df.columns:
        return

    plt.figure(figsize=(10, 6))
    plt.bar(df["name"], df["accuracy"])
    plt.xlabel("method")
    plt.ylabel("accuracy")
    plt.title("Baseline Accuracy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_exhaustive_action_scores(exhaustive_path: str, output_path: str, top_k: int = 15) -> None:
    result = load_json(exhaustive_path)
    scores = result.get("all_action_scores", [])
    if not scores:
        return

    df = pd.DataFrame(scores)
    df = df.sort_values(["avg_accuracy", "avg_reward"], ascending=False).head(top_k)

    plt.figure(figsize=(12, 6))
    plt.bar(df["action_idx"].astype(str), df["avg_accuracy"])
    plt.xlabel("action_idx")
    plt.ylabel("avg_accuracy")
    plt.title(f"Top-{top_k} Exhaustive Action Scores")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_analysis_plots(cfg) -> None:
    train_history_path = os.path.join(cfg.output_dir, cfg.train_log_json)
    leaderboard_path = os.path.join(cfg.output_dir, cfg.leaderboard_json)
    exhaustive_path = os.path.join(cfg.output_dir, cfg.exhaustive_json)
    rl_policy_path = os.path.join(cfg.output_dir, cfg.rl_policy_json)

    if os.path.exists(train_history_path):
        plot_train_curves(
            train_history_path=train_history_path,
            output_path=os.path.join(cfg.output_dir, cfg.train_curve_png),
        )

    if os.path.exists(leaderboard_path):
        plot_baseline_accuracy(
            leaderboard_path=leaderboard_path,
            output_path=os.path.join(cfg.output_dir, cfg.baseline_bar_png),
        )

    if os.path.exists(exhaustive_path):
        plot_exhaustive_action_scores(
            exhaustive_path=exhaustive_path,
            output_path=os.path.join(cfg.output_dir, cfg.exhaustive_action_png),
        )

    if os.path.exists(rl_policy_path):
        rl_policy = load_json(rl_policy_path)
        action_hist = rl_policy.get("action_hist", {})
        if action_hist:
            plot_action_hist(
                action_hist=action_hist,
                output_path=os.path.join(cfg.output_dir, cfg.action_hist_png),
                title="RL Policy Action Histogram",
            )