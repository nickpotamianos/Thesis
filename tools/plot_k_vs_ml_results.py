#!/usr/bin/env python3
"""Generate a thesis-ready visualization comparing k=1 vs k=2 and ML impact.

The plot highlights two core narratives requested for the thesis:
1. Single-tracker operation (k=1) underperforms dual-tracker (k=2).
2. The BiasNet+FusionNet ML stack reduces error relative to the baseline tracker.

It consumes the consolidated cross-validation summary written by thesis_run_all.ipynb:
    * cv_results_comprehensive_k1_k2.csv
and stores figures under the most recent runs/*_notebook_cv_eval directory unless
an explicit --output path is provided.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def detect_latest_run(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("*_notebook_cv_eval"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No run directories found under {run_root} matching '*_notebook_cv_eval'.")
    return candidates[-1]


def build_dataframe(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"Required CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_cols = {"scenario", "Method", "Budget_K", "RMSE_3D_Mean"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {csv_path}: {sorted(missing)}")

    label_map: Dict[str, str] = {
        "Baseline": "Baseline (no ML)",
        "BiasNet+FusionNet": "BiasNet+FusionNet (ML)",
    }

    df["method_label"] = df["Method"].map(label_map).fillna(df["Method"])
    df["k_label"] = df["Budget_K"].map({1: "k=1 (single tracker)", 2: "k=2 (dual trackers)"})

    # Filter to entries we care about (Baseline and BiasNet+FusionNet variants only).
    mask = df["Method"].isin(label_map.keys())
    focused = df[mask].copy()
    if focused.empty:
        raise SystemExit("Filtered dataframe is empty; confirm the CSV contains expected methods.")

    # Sort for consistent plotting order: Baseline first, then ML; within each, k=1 then k=2.
    focused["method_order"] = focused["method_label"].map({
        "Baseline (no ML)": 0,
        "BiasNet+FusionNet (ML)": 1,
    })
    focused["k_order"] = focused["Budget_K"].map({1: 0, 2: 1})
    focused.sort_values(["method_order", "k_order"], inplace=True)
    focused.reset_index(drop=True, inplace=True)

    return focused


def create_plot(df: pd.DataFrame, output: Path, dpi: int = 300) -> None:
    methods: List[str] = df["method_label"].unique().tolist()
    k_labels = ["k=1 (single tracker)", "k=2 (dual trackers)"]
    colors = {
        "k=1 (single tracker)": "#d95f02",
        "k=2 (dual trackers)": "#1b9e77",
    }

    x = np.arange(len(methods))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8.0, 4.8))

    for idx, k_label in enumerate(k_labels):
        subset = df[df["k_label"] == k_label]
        rmse_values = subset["RMSE_3D_Mean"].to_numpy()
        offsets = x + (idx - 0.5) * width
        bar = ax.bar(offsets, rmse_values, width=width, label=k_label, color=colors[k_label])

        # Add numeric labels on each bar to emphasise the difference.
        for rect in bar:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.015,
                f"{height:.3f} m",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    # Annotate the ML improvement explicitly.
    baseline_k2 = df[(df["method_label"] == "Baseline (no ML)") & (df["Budget_K"] == 2)]["RMSE_3D_Mean"].iloc[0]
    ml_k2 = df[(df["method_label"] == "BiasNet+FusionNet (ML)") & (df["Budget_K"] == 2)]["RMSE_3D_Mean"].iloc[0]
    improvement = baseline_k2 - ml_k2
    ax.annotate(
        f"ML gain: −{improvement:.3f} m",
        xy=(x[1], ml_k2),
        xytext=(x[1] + 0.35, ml_k2 + 0.25),
        arrowprops=dict(arrowstyle="->", color="#555555"),
        fontsize=10,
        ha="left",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=11)
    ax.set_ylabel("RMSE$_{3D}$ (m)", fontsize=11)
    ax.set_title("Dual trackers and ML reduce 3D tracking error", fontsize=13, pad=12)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    subtitle = (
        "Single tracker (k=1) consistently underperforms the dual-tracker setup,\n"
        "and the BiasNet+FusionNet ML pipeline further lowers RMSE when k=2."
    )
    fig.text(0.01, -0.02, subtitle, fontsize=9)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    print(f"[OK] Saved figure to {output}")

    pdf_path = output.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"[OK] Saved companion PDF to {pdf_path}")

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (defaults to CWD).")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Specific notebook run directory; auto-detected when omitted.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for the PNG output. Defaults to <run-root>/figures/k_vs_ml_comparison.png.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()

    run_root = args.run_root.resolve() if args.run_root else detect_latest_run(repo_root / "runs")
    csv_path = repo_root / "cv_results_comprehensive_k1_k2.csv"

    df = build_dataframe(csv_path)

    output = args.output
    if output is None:
        output = run_root / "figures" / "k_vs_ml_comparison.png"

    create_plot(df, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
