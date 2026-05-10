import os
import subprocess
import sys

from configs import TrainConfig


def run_step(script_name: str):
    print(f"\n[RUN] {script_name}")
    result = subprocess.run([sys.executable, script_name], check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Step failed: {script_name}")


def main():
    cfg = TrainConfig()

    steps = [
        "run_expand_oracle_dataset.py",
        "run_build_preference_dataset_from_expanded_oracle.py",
        "run_train_action_preference_model_expanded.py",
        "run_preference_controller_holdout.py",
        "run_compare_policy_summaries.py",
    ]

    for step in steps:
        if not os.path.exists(step):
            print(f"[SKIP] missing file: {step}")
            continue
        run_step(step)

    print("\n[INFO] Full preference pipeline finished")


if __name__ == "__main__":
    main()