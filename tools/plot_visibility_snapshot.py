#!/usr/bin/env python3
"""Plot a single-timestamp snapshot of tracker geometry and suggested moves."""
from __future__ import annotations

import argparse
import math
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from miluv.data import DataLoader


def _infer_experiment(run_dir: Path) -> str:
    name = run_dir.name
    # Expected pattern: <exp>_<target>
    if "_ifo" in name:
        return name.split("_ifo", 1)[0]
    for stem in run_dir.parts[::-1]:
        if stem.startswith("default_"):
            return stem
    return name


def _select_timestamp(run_dir: Path, timestamp: float | None, window: float) -> Tuple[float, pd.DataFrame, Tuple[float, float]]:
    actions_path = run_dir / "action_suggestions.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing action_suggestions.csv in {run_dir}")
    actions = pd.read_csv(actions_path)
    actions["norm"] = np.linalg.norm(actions[["dx", "dy", "dz"]].to_numpy(), axis=1)
    if timestamp is not None:
        idx = (actions["timestamp"] - timestamp).abs().idxmin()
        ts = float(actions.loc[idx, "timestamp"])
        subset = actions.copy()
    else:
        eig_path = run_dir / "planner_eig_scores.csv"
        ts = None
        subset = actions
        if eig_path.exists():
            eig = pd.read_csv(eig_path)
            if not eig.empty:
                idx = eig["eig_score"].idxmax()
                ts = float(eig.loc[idx, "timestamp"])
        if ts is None:
            idx = actions["norm"].idxmax()
            ts = float(actions.loc[idx, "timestamp"])

    if window > 0:
        half = window / 2.0
        mask = (subset["timestamp"] >= ts - half) & (subset["timestamp"] <= ts + half)
        snap = subset[mask].copy()
        t_min = float(snap["timestamp"].min()) if not snap.empty else ts
        t_max = float(snap["timestamp"].max()) if not snap.empty else ts
    else:
        snap = subset[subset["timestamp"] == ts].copy()
        t_min = t_max = ts

    if snap.empty:
        snap = subset[subset["timestamp"] == ts].copy()
        t_min = t_max = ts

    if window > 0 and not snap.empty:
        agg = snap.groupby("tracker").agg({"dx": "mean", "dy": "mean", "dz": "mean", "norm": "mean", "timestamp": "count"}).reset_index()
        agg = agg.rename(columns={"timestamp": "samples", "norm": "norm_mean"})
    else:
        agg = snap.copy()
        agg["samples"] = 1
    return ts, agg, (t_min, t_max)



def _load_positions(exp: str, timestamp: float) -> Dict[str, np.ndarray]:
    loader = DataLoader(
        exp,
        exp_dir="./data/three_robots",
        cir=False,
        barometer=False,
        height=True,
        imu="px4",
        cam=None,
        mag=False,
    )
    positions: Dict[str, np.ndarray] = {}
    for robot, sensors in loader.data.items():
        pos_fn = sensors["mocap_pos"]
        pos = np.asarray(pos_fn([timestamp]), dtype=float)
        if pos.ndim == 2 and pos.shape[1] == 1:
            positions[robot] = pos[:, 0]
        else:
            positions[robot] = pos.reshape(-1)[-3:]
    return positions


def _plot_snapshot(run_dir: Path, roles: dict, ts: float, snapshot: pd.DataFrame, out_path: Path, window_range: Tuple[float, float]) -> None:
    exp = _infer_experiment(run_dir)
    positions = _load_positions(exp, ts)
    target = roles["target"]
    trackers = roles["trackers"]

    p_target = positions[target]
    tracker_points = {trk: positions[trk] for trk in trackers}

    fig = plt.figure(figsize=(12, 5))
    ax_xy = fig.add_subplot(1, 2, 1)
    ax_xz = fig.add_subplot(1, 2, 2)

    colors = {trk: plt.cm.tab10(i) for i, trk in enumerate(trackers)}

    # Plot target position
    ax_xy.scatter(p_target[0], p_target[1], marker="*", color="black", s=200, label="Target")
    ax_xz.scatter(p_target[0], p_target[2], marker="*", color="black", s=200, label="Target")

    annotations = []
    for trk in trackers:
        p = tracker_points[trk]
        row = snapshot[snapshot["tracker"] == trk]
        if row.empty:
            continue
        dx, dy, dz = row[["dx", "dy", "dz"]].iloc[0]
        cmd = np.array([dx, dy, dz], dtype=float)
        toward = p_target - p
        dist = float(np.linalg.norm(toward))
        ez = abs(toward[2] / dist) if dist > 1e-6 else float("nan")
        annotations.append((trk, dist, ez, cmd))

        # Top-down view arrows
        ax_xy.scatter(p[0], p[1], color=colors[trk], s=80, label=trk)
        ax_xy.annotate(trk, (p[0], p[1]), textcoords="offset points", xytext=(5, 5))
        ax_xy.arrow(
            p[0], p[1], cmd[0], cmd[1],
            color=colors[trk],
            width=0.01,
            head_width=0.2,
            length_includes_head=True,
            alpha=0.7,
        )
        # 3D command projected into XZ
        ax_xz.scatter(p[0], p[2], color=colors[trk], s=80, label=trk)
        ax_xz.annotate(trk, (p[0], p[2]), textcoords="offset points", xytext=(5, 5))
        ax_xz.arrow(
            p[0], p[2], cmd[0], cmd[2],
            color=colors[trk],
            width=0.01,
            head_width=0.2,
            length_includes_head=True,
            alpha=0.7,
        )
        # Draw LOS line
        ax_xy.plot([p[0], p_target[0]], [p[1], p_target[1]], linestyle="--", color=colors[trk], alpha=0.4)
        ax_xz.plot([p[0], p_target[0]], [p[2], p_target[2]], linestyle="--", color=colors[trk], alpha=0.4)

    ax_xy.set_title(f"Top view @ t={ts:.2f}s")
    ax_xy.set_xlabel("x [m]")
    ax_xy.set_ylabel("y [m]")
    ax_xy.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.legend()

    ax_xz.set_title("Elevation view")
    ax_xz.set_xlabel("x [m]")
    ax_xz.set_ylabel("z [m]")
    ax_xz.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax_xz.set_aspect("equal", adjustable="box")

    # Text box summarizing geometry
    data_rows = []
    for trk, dist, ez, cmd in annotations:
        samples = int(snapshot[snapshot["tracker"] == trk]["samples"].iloc[0]) if "samples" in snapshot.columns else 1
        data_rows.append([trk, dist, ez, cmd[0], cmd[1], cmd[2], samples])
    columns = ["Tracker", "Range (m)", "|e_z|", "Δx", "Δy", "Δz", "Samples"]
    df = pd.DataFrame(data_rows, columns=columns)
    df_display = df.to_string(index=False, float_format=lambda v: f"{v:6.2f}")
    planner = "EIG" if any("EIG" in part or "eig" in part.lower() for part in run_dir.parts) else "heuristic"
    if window_range[0] == window_range[1]:
        time_info = f"Timestamp: {ts:.2f} s"
    else:
        time_info = f"Window: [{window_range[0]:.2f}, {window_range[1]:.2f}] s"
    caption = f"Experiment: {exp}\nPlanner: {planner}\n{time_info}"
    fig.text(0.5, 0.01, df_display, ha="center", va="bottom", family="monospace")
    fig.text(0.02, 0.01, caption, ha="left", va="bottom")
    print(caption)
    print(df_display)

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a visibility snapshot from a run directory.")
    parser.add_argument("run_dir", type=Path, help="Run directory containing roles.json and action_suggestions.csv")
    parser.add_argument("--timestamp", type=float, default=None, help="Timestamp to visualize (seconds)")
    parser.add_argument("--out", type=Path, default=None, help="Output image path")
    parser.add_argument("--window", type=float, default=0.0, help="Averaging window in seconds centered on timestamp")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    roles_path = run_dir / "roles.json"
    if not roles_path.exists():
        raise FileNotFoundError(f"Missing roles.json in {run_dir}")
    with roles_path.open("r") as f:
        roles_dict = json.load(f)

    ts, snapshot, window_range = _select_timestamp(run_dir, args.timestamp, args.window)
    if snapshot.empty:
        raise ValueError("No action suggestions found for selected timestamp")

    out_path = args.out.resolve() if args.out else run_dir / f"visibility_snapshot_{ts:.2f}s.png"
    _plot_snapshot(run_dir, roles_dict, ts, snapshot, out_path, window_range)
    print("Saved snapshot to:", out_path)


if __name__ == "__main__":
    main()
