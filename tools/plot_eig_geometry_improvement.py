#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from miluv.data import DataLoader


def _load_commands(csv_path: Path, window_start: float, window_end: float) -> Dict[str, np.ndarray]:
    df = pd.read_csv(csv_path)
    if not {"timestamp", "tracker", "dx", "dy", "dz"}.issubset(df.columns):
        missing = {"timestamp", "tracker", "dx", "dy", "dz"} - set(df.columns)
        raise ValueError(f"Missing columns {missing} in {csv_path}")
    mask = (df["timestamp"] >= window_start) & (df["timestamp"] <= window_end)
    if not mask.any():
        raise ValueError(f"No action rows within [{window_start}, {window_end}] for {csv_path}")
    means = df.loc[mask].groupby("tracker")[["dx", "dy", "dz"]].mean()
    return {tracker: row.values for tracker, row in means.iterrows()}


def _get_positions(loader: DataLoader, timestamp: float, trackers: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    positions = {}
    for tracker in trackers:
        data = loader.data[tracker]["mocap_pos"]([timestamp])
        if data.size == 0:
            raise ValueError(f"No mocap data for {tracker} at t={timestamp}")
        positions[tracker] = data[:, 0]
    return positions


def _compute_metrics(target: np.ndarray, plan_positions: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    metrics: Dict[str, Dict[str, float]] = {}
    trackers = list(plan_positions.keys())
    for tracker, pos in plan_positions.items():
        diff = target - pos
        rng = float(np.linalg.norm(diff))
        metrics[tracker] = {
            "range": rng,
            "vert_ratio": float(abs(diff[2]) / rng),
        }
    if len(trackers) >= 2:
        baseline = float(np.linalg.norm(plan_positions[trackers[0]] - plan_positions[trackers[1]]))
        for tracker in trackers:
            metrics[tracker]["baseline"] = baseline
    return metrics


def _plot_top_view(ax, current_positions, plan_positions, target, trackers, plan_styles):
    marker_map = ["o", "s", "^", "D", "P"]
    ax.scatter(target[0], target[1], marker="*", s=140, color="black", label="Target")
    current_handles = {}
    for idx, tracker in enumerate(trackers):
        marker = marker_map[idx % len(marker_map)]
        pos = current_positions[tracker]
        handle = ax.scatter(pos[0], pos[1], color="#4f4f4f", marker=marker, s=60)
        current_handles[tracker] = (handle, marker)
        ax.text(pos[0], pos[1], f" {tracker}", fontsize=9, color="#3a3a3a")
    for plan_label, style in plan_styles.items():
        positions = plan_positions[plan_label]
        for tracker in trackers:
            start = current_positions[tracker]
            end = positions[tracker]
            dx, dy = end[0] - start[0], end[1] - start[1]
            if np.hypot(dx, dy) < 1e-6:
                continue
            ax.arrow(
                start[0],
                start[1],
                dx,
                dy,
                color=style["color"],
                width=0.01,
                head_width=0.15,
                length_includes_head=True,
                linestyle=style.get("linestyle", "-"),
                alpha=0.9,
                label=style.get("label"),
            )
            ax.scatter(
                end[0],
                end[1],
                color=style["color"],
                marker=current_handles[tracker][1],
                s=60,
                edgecolor="white",
                zorder=3,
            )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Top-down geometry")
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc="best")


def _plot_metric_bars(ax, trackers, plan_labels, metric_data, metric_key, ylabel):
    x = np.arange(len(trackers))
    width = 0.25
    for idx, plan_label in enumerate(plan_labels):
        values = [metric_data[plan_label][tracker][metric_key] for tracker in trackers]
        ax.bar(x + idx * width - width, values, width=width, label=plan_label)
    ax.set_xticks(x)
    ax.set_xticklabels(trackers)
    ax.set_ylabel(ylabel)
    ax.set_title(metric_key.replace("_", " ").capitalize())
    ax.legend()


def main():
    parser = argparse.ArgumentParser(description="Plot EIG vs heuristic geometry around a timestamp.")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--timestamp", type=float, required=True)
    parser.add_argument("--window-start", type=float, required=True)
    parser.add_argument("--window-end", type=float, required=True)
    parser.add_argument("--eig-actions", type=Path, required=True)
    parser.add_argument("--heur-actions", type=Path, required=True)
    parser.add_argument("--trackers", default="ifo001,ifo002")
    parser.add_argument("--data-root", type=Path, default=Path("./data/three_robots"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="EIG action geometry improvement")
    args = parser.parse_args()

    trackers = [trk.strip() for trk in args.trackers.split(",") if trk.strip()]
    if len(trackers) < 2:
        raise ValueError("At least two trackers are required.")

    eig_commands = _load_commands(args.eig_actions, args.window_start, args.window_end)
    heur_commands = _load_commands(args.heur_actions, args.window_start, args.window_end)
    if set(trackers) - eig_commands.keys():
        missing = set(trackers) - eig_commands.keys()
        raise ValueError(f"Missing EIG commands for trackers: {sorted(missing)}")
    if set(trackers) - heur_commands.keys():
        missing = set(trackers) - heur_commands.keys()
        raise ValueError(f"Missing heuristic commands for trackers: {sorted(missing)}")

    loader = DataLoader(
        args.experiment,
        exp_dir=str(args.data_root),
        height=True,
        imu="px4",
        cir=False,
        barometer=False,
        cam=None,
        mag=False,
    )

    current_positions = _get_positions(loader, args.timestamp, {trk: None for trk in trackers})
    target = loader.data["ifo003"]["mocap_pos"]([args.timestamp])
    if target.size == 0:
        raise ValueError("No target mocap data at the requested timestamp.")
    target = target[:, 0]

    plan_commands = {
        "Current": {tracker: np.zeros(3) for tracker in trackers},
        "Heuristic": heur_commands,
        "EIG": eig_commands,
    }
    plan_positions = {}
    plan_metrics = {}
    for plan_label, commands in plan_commands.items():
        positions = {}
        for tracker in trackers:
            positions[tracker] = current_positions[tracker] + commands[tracker]
        plan_positions[plan_label] = positions
        plan_metrics[plan_label] = _compute_metrics(target, positions)

    plan_styles = {
        "Heuristic": {"color": "#ff7f0e", "linestyle": "--", "label": "Heuristic command"},
        "EIG": {"color": "#1f77b4", "linestyle": "-", "label": "EIG command"},
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    _plot_top_view(axes[0], current_positions, plan_positions, target, trackers, plan_styles)
    _plot_metric_bars(axes[1], trackers, ["Current", "Heuristic", "EIG"], plan_metrics, "range", "Range to target [m]")
    _plot_metric_bars(axes[2], trackers, ["Current", "Heuristic", "EIG"], plan_metrics, "vert_ratio", "|e_z| (unitless)")

    baseline_text = "\n".join(
        f"{plan}: baseline = {plan_metrics[plan][trackers[0]]['baseline']:.2f} m"
        for plan in ["Current", "Heuristic", "EIG"]
    )
    axes[2].text(0.5, -0.35, baseline_text, transform=axes[2].transAxes, ha="center", fontsize=9)

    fig.suptitle(args.title)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
