import os

from configs import TrainConfig
from src.analysis import generate_all_analysis_plots
from src.utils import ensure_dir, print_artifact_summary


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    generate_all_analysis_plots(cfg)

    print("[INFO] analysis plots generated")
    print_artifact_summary(
        "saved artifacts",
        {
            "output_dir": cfg.output_dir,
            "train_curve": os.path.join(cfg.output_dir, cfg.train_curve_png),
            "action_hist": os.path.join(cfg.output_dir, cfg.action_hist_png),
            "baseline_bar": os.path.join(cfg.output_dir, cfg.baseline_bar_png),
            "exhaustive_action": os.path.join(cfg.output_dir, cfg.exhaustive_action_png),
        },
    )


if __name__ == "__main__":
    main()
