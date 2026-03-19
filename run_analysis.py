from configs import TrainConfig
from src.analysis import generate_all_analysis_plots
from src.utils import ensure_dir


def main():
    cfg = TrainConfig()
    ensure_dir(cfg.output_dir)

    generate_all_analysis_plots(cfg)

    print("[INFO] analysis plots generated")
    print(f"[INFO] output_dir={cfg.output_dir}")
    print(f"[INFO] train_curve={cfg.train_curve_png}")
    print(f"[INFO] action_hist={cfg.action_hist_png}")
    print(f"[INFO] baseline_bar={cfg.baseline_bar_png}")
    print(f"[INFO] exhaustive_action={cfg.exhaustive_action_png}")


if __name__ == "__main__":
    main()