#!/usr/bin/env python3
"""Utility to summarize and visualize smart-tracker action suggestions and EIG diagnostics."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _bin_actions(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    """Down-sample large action logs by averaging in time bins."""
    if df.empty:
        return df.copy()
    # Bin timestamps to keep plotting manageable while preserving trends.
    t = df["timestamp"].to_numpy(dtype=float)
    bins = np.floor(t / max(bin_width, 1e-6)) * bin_width
    df = df.assign(time_bin=bins)
    grouped = (
        df.groupby(["tracker", "time_bin"], as_index=False)[["dx", "dy", "dz", "norm"]]
        .mean()
        .rename(columns={"time_bin": "timestamp"})
    )
    return grouped


def _plot_component_timeseries(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    components = ["dx", "dy", "dz", "norm"]
    titles = ["Δx", "Δy", "Δz", "Step norm"]
    for ax, comp, title in zip(axes, components, titles):
        for tracker, grp in df.groupby("tracker"):
            ax.plot(grp["timestamp"], grp[comp], label=tracker, linewidth=1.0, alpha=0.85)
        ax.set_ylabel(title)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    axes[-1].set_xlabel("Timestamp [s]")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_xy_quiver(df: pd.DataFrame, out_path: Path, max_points: int = 500) -> None:
    trackers = sorted(df["tracker"].unique())
    n = len(trackers)
    ncols = min(3, n) if n > 0 else 1
    nrows = math.ceil(n / ncols) if n > 0 else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    for ax in axes_flat:
        ax.axis("off")
    for ax, tracker in zip(axes_flat, trackers):
        subset = df[df["tracker"] == tracker]
        if max_points and len(subset) > max_points:
            subset = subset.sample(max_points, random_state=7)
        ax.axis("on")
        ax.quiver(
            np.zeros(len(subset)),
            np.zeros(len(subset)),
            subset["dx"],
            subset["dy"],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            alpha=0.4,
            color="#1f77b4",
        )
        ax.set_title(f"{tracker} XY suggestions")
        ax.set_xlabel("Δx [m]")
        ax.set_ylabel("Δy [m]")
        ax.set_xlim(-0.45, 0.45)
        ax.set_ylim(-0.45, 0.45)
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        ax.set_aspect("equal", adjustable="box")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_eig_diagnostics(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(10, 4))
    for tracker, grp in df.groupby("tracker"):
        ax1.plot(grp["timestamp"], grp["eig_score"], label=f"{tracker} EIG", linewidth=1.0)
    ax1.set_xlabel("Timestamp [s]")
    ax1.set_ylabel("Expected Δtrace")
    ax1.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax1.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize_actions(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "dx": ["mean", "std", "min", "max", "median"],
        "dy": ["mean", "std", "min", "max", "median"],
        "dz": ["mean", "std", "min", "max", "median"],
        "norm": ["mean", "std", "min", "max", "median"],
    }
    stats = df.groupby("tracker").agg(agg)
    stats.columns = [f"{comp}_{metric}" for comp, metric in stats.columns]
    # Add decile information for norm to capture spread succinctly.
    deciles = df.groupby("tracker")["norm"].quantile([0.1, 0.9]).unstack(level=1)
    deciles = deciles.rename(columns={0.1: "norm_p10", 0.9: "norm_p90"})
    stats = stats.join(deciles)
    return stats


def summarize_eig(df: pd.DataFrame) -> pd.DataFrame:
    stats = (
        df.groupby("tracker")[["eig_score", "R_eff"]]
        .agg(["mean", "std", "min", "max", "median"])
    )
    stats.columns = [f"{col}_{agg}" for col, agg in stats.columns]
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize action suggestions and EIG diagnostics for a run.")
    parser.add_argument("run_dir", type=Path, help="Path to run directory containing action_suggestions.csv")
    parser.add_argument("--bin-width", type=float, default=1.0, help="Seconds per aggregation bin for plotting")
    parser.add_argument("--sample", type=int, default=0, help="Randomly subsample this many rows before binning (0=all)")
    parser.add_argument("--out", type=Path, default=None, help="Directory to save plots and tables (defaults to run_dir)")
    args = parser.parse_args()

    run_dir = args.run_dir
    out_dir = args.out if args.out is not None else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    actions_path = run_dir / "action_suggestions.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"No action_suggestions.csv found under {run_dir}")

    df_actions = pd.read_csv(actions_path)
    df_actions["norm"] = np.linalg.norm(df_actions[["dx", "dy", "dz"]].to_numpy(), axis=1)
    if args.sample and len(df_actions) > args.sample:
        df_actions = df_actions.sample(args.sample, random_state=7).sort_values("timestamp")

    summary = summarize_actions(df_actions)
    summary.to_csv(out_dir / "action_suggestions_summary.csv")

    binned = _bin_actions(df_actions, args.bin_width)
    _plot_component_timeseries(binned, out_dir / "action_components_timeseries.png")
    _plot_xy_quiver(df_actions, out_dir / "action_xy_quiver.png")

    eig_path = run_dir / "planner_eig_scores.csv"
    if eig_path.exists():
        df_eig = pd.read_csv(eig_path)
        eig_summary = summarize_eig(df_eig)
        eig_summary.to_csv(out_dir / "planner_eig_summary.csv")
        _plot_eig_diagnostics(df_eig, out_dir / "planner_eig_timeseries.png")
    else:
        eig_summary = None

    print("Saved action summary to:", out_dir / "action_suggestions_summary.csv")
    print("Saved plots:")
    print(" -", out_dir / "action_components_timeseries.png")
    print(" -", out_dir / "action_xy_quiver.png")
    if eig_summary is not None:
        print(" -", out_dir / "planner_eig_timeseries.png")


if __name__ == "__main__":
    main()
