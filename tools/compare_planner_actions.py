#!/usr/bin/env python3
"""Compare heuristic vs. EIG planner action suggestions for a given experiment."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _load_actions(run_dir: Path, label: str) -> pd.DataFrame:
    actions_path = run_dir / "action_suggestions.csv"
    if not actions_path.exists():
        raise FileNotFoundError(f"{label}: missing action_suggestions.csv under {run_dir}")
    df = pd.read_csv(actions_path)
    df["norm"] = np.linalg.norm(df[["dx", "dy", "dz"]].to_numpy(), axis=1)
    df["planner"] = label
    return df


def _bin_norms(df: pd.DataFrame, bin_width: float) -> pd.DataFrame:
    if df.empty:
        return df
    bins = np.floor(df["timestamp"].to_numpy(dtype=float) / max(bin_width, 1e-6)) * bin_width
    out = df.assign(time_bin=bins)
    grouped = (
        out.groupby(["planner", "tracker", "time_bin"], as_index=False)["norm"]
        .mean()
        .rename(columns={"time_bin": "timestamp", "norm": "norm_mean"})
    )
    return grouped


def _planner_palette(labels: list[str]) -> dict[str, str]:
    base = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    return {lbl: base[i % len(base)] for i, lbl in enumerate(labels)}


def _plot_norm_timeseries(binned: pd.DataFrame, out_path: Path) -> None:
    planners = binned["planner"].unique().tolist()
    palette = _planner_palette(planners)
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped = binned.groupby(["planner", "timestamp"], as_index=False)["norm_mean"].mean()
    for planner, sub in grouped.groupby("planner"):
        ax.plot(sub["timestamp"], sub["norm_mean"], label=planner, color=palette[planner], linewidth=1.2)
    ax.set_xlabel("Timestamp [s]")
    ax.set_ylabel("Mean step norm [m]")
    ax.set_title("Planner step magnitude over time")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_norm_hist(df: pd.DataFrame, out_path: Path) -> None:
    planners = df["planner"].unique().tolist()
    palette = _planner_palette(planners)
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.linspace(0.0, 0.42, 30)
    for planner in planners:
        subset = df[df["planner"] == planner]
        ax.hist(
            subset["norm"],
            bins=bins,
            density=True,
            histtype="step",
            linewidth=1.4,
            label=planner,
            color=palette[planner],
        )
    ax.set_xlabel("Step norm [m]")
    ax.set_ylabel("Density")
    ax.set_title("Distribution of action magnitudes")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_eig_timeseries(eig_df: pd.DataFrame, out_path: Path) -> None:
    if eig_df.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 4))
    for tracker, sub in eig_df.groupby("tracker"):
        ax.plot(sub["timestamp"], sub["eig_score"], label=f"{tracker} EIG", linewidth=1.1)
    ax.set_xlabel("Timestamp [s]")
    ax.set_ylabel("Expected trace reduction")
    ax.set_title("EIG planner diagnostics")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_eig(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "planner_eig_scores.csv"
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "tracker", "eig_score", "R_eff"])
    return pd.read_csv(path)


def _summarize(df: pd.DataFrame) -> pd.DataFrame:
    agg = {
        "norm": ["mean", "std", "median", "min", "max"],
        "dx": ["mean", "std"],
        "dy": ["mean", "std"],
        "dz": ["mean", "std"],
    }
    summary = df.groupby(["planner", "tracker"]).agg(agg)
    summary.columns = [f"{component}_{metric}" for component, metric in summary.columns]
    q = df.groupby(["planner", "tracker"])["norm"].quantile([0.1, 0.9]).unstack(level=2)
    q = q.rename(columns={0.1: "norm_p10", 0.9: "norm_p90"})
    summary = summary.join(q)
    return summary.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare action suggestions between planners.")
    parser.add_argument("baseline", type=Path, help="Run directory for heuristic planner output")
    parser.add_argument("eig", type=Path, help="Run directory for EIG planner output")
    parser.add_argument("--bin-width", type=float, default=2.0, help="Seconds per bin for timeseries plots")
    parser.add_argument("--sample", type=int, default=0, help="Optional random sample size per planner for histograms")
    parser.add_argument("--out", type=Path, default=None, help="Directory for comparative plots (default: baseline parent)")
    args = parser.parse_args()

    if args.out is not None:
        out_dir = args.out.resolve()
    else:
        out_dir = (args.baseline.parent / "comparison_plots").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = _load_actions(args.baseline, "heuristic")
    eig_df = _load_actions(args.eig, "EIG")
    combined = pd.concat([baseline_df, eig_df], ignore_index=True)

    if args.sample and len(combined) > 2 * args.sample:
        sampled = []
        for planner, sub in combined.groupby("planner"):
            if len(sub) > args.sample:
                sampled.append(sub.sample(args.sample, random_state=7))
            else:
                sampled.append(sub)
        combined = pd.concat(sampled, ignore_index=True)

    summary = _summarize(combined)
    summary_path = out_dir / "planner_action_comparison.csv"
    summary.to_csv(summary_path, index=False)

    binned = _bin_norms(combined, args.bin_width)
    _plot_norm_timeseries(binned, out_dir / "planner_norm_timeseries.png")
    _plot_norm_hist(combined, out_dir / "planner_norm_hist.png")

    eig_scores = _load_eig(args.eig)
    if not eig_scores.empty:
        _plot_eig_timeseries(eig_scores, out_dir / "planner_eig_timeseries.png")

    print("Saved comparative summary to:", summary_path)
    print("Saved plots:")
    print(" -", out_dir / "planner_norm_timeseries.png")
    print(" -", out_dir / "planner_norm_hist.png")
    if not eig_scores.empty:
        print(" -", out_dir / "planner_eig_timeseries.png")


if __name__ == "__main__":
    main()
