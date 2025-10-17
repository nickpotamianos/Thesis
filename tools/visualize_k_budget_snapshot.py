#!/usr/bin/env python3
"""Visualize tracker selection under top-k budgeting using a logged run snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

try:
    import matplotlib.pyplot as plt  # type: ignore[import]
except ImportError as exc:
    raise SystemExit("matplotlib is required to run this script.") from exc

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Run snapshot directory containing fusion_weights.csv and roles.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output path for the generated figure (PNG).",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Downsample the time series to this many points for plotting.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-3,
        help="Minimum fusion weight for a tracker to be counted as active.",
    )
    return parser.parse_args()


def load_roles(run_dir: Path) -> List[str]:
    roles_path = run_dir / "roles.json"
    if not roles_path.exists():
        return []
    with roles_path.open() as fh:
        payload: Dict[str, object] = json.load(fh)
    trackers = payload.get("trackers", [])
    return [str(t) for t in trackers]


def load_weights(run_dir: Path, threshold: float) -> tuple[pd.DataFrame, List[str]]:
    weights_path = run_dir / "fusion_weights.csv"
    if not weights_path.exists():
        raise SystemExit(f"Missing fusion weights: {weights_path}")

    df = pd.read_csv(weights_path)
    if df.empty:
        raise SystemExit("fusion_weights.csv is empty; nothing to visualize.")
    if "timestamp" not in df.columns:
        raise SystemExit("fusion_weights.csv missing 'timestamp' column.")

    weight_cols = [col for col in df.columns if col.startswith("w_")]
    if not weight_cols:
        raise SystemExit("No tracker weight columns found (expected columns named 'w_<tracker>').")

    df = df.sort_values("timestamp").reset_index(drop=True)
    df["elapsed_s"] = df["timestamp"] - df["timestamp"].iloc[0]
    df["active_count"] = (df[weight_cols] > threshold).sum(axis=1)
    df["primary_tracker"] = (
        df[weight_cols]
        .idxmax(axis=1)
        .str.replace("^w_", "", regex=True)
    )
    return df, weight_cols


def downsample(df: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if max_points <= 0 or len(df) <= max_points:
        return df
    step = max(1, len(df) // max_points)
    return df.iloc[::step].copy()


def resolve_output_path(run_dir: Path, output: Path | None) -> Path:
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir / "k_budget_snapshot.png"


def create_plot(
    df: pd.DataFrame,
    weight_cols: List[str],
    trackers: List[str],
    output_path: Path,
    observed_cap: int,
    threshold: float,
    share_lines: str,
) -> None:
    df_plot = df
    times = df_plot["elapsed_s"].to_numpy()
    series = [df_plot[col].to_numpy() for col in weight_cols]
    labels = [col[2:] if col.startswith("w_") else col for col in weight_cols]

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(len(weight_cols))]

    fig, (ax_weights, ax_active) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

    ax_weights.stackplot(times, series, labels=labels, colors=colors, alpha=0.85)
    ax_weights.set_ylabel("Fusion weight")
    ax_weights.set_title("Tracker weights after budgeting", pad=10)
    ax_weights.set_xlim(times[0], times[-1])
    ax_weights.set_ylim(0.0, 1.05)
    ax_weights.legend(loc="upper right", frameon=False, ncol=max(1, len(labels) // 3))
    ax_weights.grid(alpha=0.3, linestyle="--")

    ax_active.step(times, df_plot["active_count"], where="post", color="#2c7fb8", linewidth=1.5)
    ax_active.set_ylabel("Active trackers")
    ax_active.set_xlabel("Elapsed time (s)")
    ax_active.set_ylim(-0.1, len(weight_cols) + 0.5)
    ax_active.grid(alpha=0.3, linestyle="--")

    ax_active.axhline(observed_cap, color="#b2182b", linestyle=":", linewidth=1.2, label="Observed cap")
    if trackers:
        ax_active.axhline(len(trackers), color="#4daf4a", linestyle="--", linewidth=1.0, label="Available trackers")
    ax_active.legend(loc="upper right", frameon=False)

    subtitle = (
        f"Threshold for active trackers: {threshold:.1e}. "
        f"Observed max active trackers: {observed_cap} of {len(weight_cols)} weight streams."
    )
    fig.text(0.01, 0.02, subtitle + "\n" + share_lines, fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")

    df_weights, weight_cols = load_weights(run_dir, args.threshold)
    trackers = load_roles(run_dir)

    observed_cap = int(df_weights["active_count"].max())
    df_plot = downsample(df_weights, args.max_points)

    shares = df_weights["primary_tracker"].value_counts(normalize=True, dropna=True)
    if shares.empty:
        share_lines = "Primary tracker share: unavailable"
    else:
        parts = [f"{name}: {pct * 100:.1f}%" for name, pct in shares.sort_index().items()]
        share_lines = "Primary tracker share: " + ", ".join(parts)

    output_path = resolve_output_path(run_dir, args.output)

    create_plot(df_plot, weight_cols, trackers, output_path, observed_cap, args.threshold, share_lines)

    print(f"[OK] Saved visualization to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
