#!/usr/bin/env python3
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RUN_ROOT = "runs/20251004_random3_repro_baseline"
EXPERIMENTS: List[Tuple[str, str]] = [
    ("default_3_random3_0b", "Random3-0b"),
    ("default_3_random3_1", "Random3-1"),
    ("default_3_random3_2", "Random3-2"),
]


def load_metrics() -> pd.DataFrame:
    records = []
    for exp_id, label in EXPERIMENTS:
        base_dir = os.path.join(RUN_ROOT, exp_id, f"{exp_id}_ifo003")
        summary_path = os.path.join(base_dir, "summary.csv")
        tracker_path = os.path.join(base_dir, "tracker_summary.csv")
        if not os.path.exists(summary_path):
            raise FileNotFoundError(f"Missing summary.csv for {exp_id} at {summary_path}")
        if not os.path.exists(tracker_path):
            raise FileNotFoundError(f"Missing tracker_summary.csv for {exp_id} at {tracker_path}")

        summary = pd.read_csv(summary_path).iloc[0]
        records.append(
            {
                "experiment": label,
                "entity": "Target ifo003",
                "rmse_x": float(summary["rmse_x"]),
                "rmse_y": float(summary["rmse_y"]),
                "rmse_z": float(summary["rmse_z"]),
                "rmse_3d": float(summary["rmse_3d"]),
            }
        )

        tracker_df = pd.read_csv(tracker_path)
        for _, row in tracker_df.iterrows():
            records.append(
                {
                    "experiment": label,
                    "entity": f"Tracker {row['tracker']}",
                    "rmse_x": float(row["rmse_x"]),
                    "rmse_y": float(row["rmse_y"]),
                    "rmse_z": float(row["rmse_z"]),
                    "rmse_3d": float(row["rmse_3d"]),
                }
            )
    return pd.DataFrame.from_records(records)


def plot_metrics(df: pd.DataFrame) -> str:
    out_dir = os.path.join(RUN_ROOT, "figures")
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, "random3_target_tracker_rmse.png")

    metrics = ["rmse_x", "rmse_y", "rmse_z", "rmse_3d"]
    metric_titles = {
        "rmse_x": "RMSE X (m)",
        "rmse_y": "RMSE Y (m)",
        "rmse_z": "RMSE Z (m)",
        "rmse_3d": "RMSE 3D (m)",
    }
    entities = ["Target ifo003", "Tracker ifo001", "Tracker ifo002"]
    colors = {
        "Target ifo003": "#1f77b4",
        "Tracker ifo001": "#ff7f0e",
        "Tracker ifo002": "#2ca02c",
    }

    x = np.arange(len(EXPERIMENTS))
    width = 0.25

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4), sharey=False)

    # Draw grouped bars per metric so that tracker vs target differences are clear.
    for ax, metric in zip(axes, metrics):
        for idx, entity in enumerate(entities):
            offset = (idx - 1) * width
            heights = [
                df[(df["entity"] == entity) & (df["experiment"] == label)][metric].values[0]
                for _, label in EXPERIMENTS
            ]
            bars = ax.bar(x + offset, heights, width=width, label=entity, color=colors[entity])
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([label for _, label in EXPERIMENTS])
        ax.set_title(metric_titles[metric])
        ax.set_ylim(bottom=0.0)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6)

    axes[0].set_ylabel("RMSE (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle("Baseline RMSE by Experiment for Target and Trackers", y=0.98)
    fig.legend(handles, labels, loc="upper center", ncol=len(entities), frameon=False, bbox_to_anchor=(0.5, 0.93))
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig_path


def main() -> None:
    df = load_metrics()
    df.sort_values(["experiment", "entity"], inplace=True)
    print("Loaded metrics:\n", df.to_string(index=False))
    fig_path = plot_metrics(df)
    print(f"Saved figure to {fig_path}")


if __name__ == "__main__":
    main()
