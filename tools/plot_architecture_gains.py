#!/usr/bin/env python3
"""Generate an extensive set of plots showing architecture gains per experiment/scenario.

Outputs (saved under the latest runs/*_notebook_cv_eval/figures directory unless --output-root is
provided):

* cv_architecture_gains_summary.(png|pdf)
    - Bar chart of RMSE_3D means for each model scenario across cross-validation.
* cv_architecture_gains_per_fold.(png|pdf)
    - Scatter/box hybrid highlighting per-fold RMSE deltas (Baseline minus model).
* zigzag_architecture_gains_per_experiment.(png|pdf)
    - Grouped bar plots comparing Baseline, BiasNet+FusionNet, and Budgeted k=2 for each zigzag LOEO experiment.
* zigzag_architecture_gains_summary.(png|pdf)
    - Bar chart showing zigzag scenario averages with best performer annotations.

The script expects the CSVs emitted by thesis_run_all.ipynb to exist at the repository root and under
runs/<timestamp> directories:
    cv_results_aggregated.csv
    cv_results_summary.csv
    runs/<timestamp>_notebook_cv_eval/zigzag_loeo_results.csv
    runs/<timestamp>_notebook_cv_eval/zigzag_loeo_summary.csv

Extend this script if additional scenarios or metrics need to be visualised.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "font.size": 11,
})

SUMMARY_SCENARIO_ORDER = [
    "Baseline_Grid",
    "BiasNet_ByTime",
    "BiasNet_ByExp",
    "FusionNet_ByTime",
    "FusionNet_ByExp",
    "Budgeted_K2",
    "BiasNet+FusionNet_ByExp",
]

PLOT_PALETTE = {
    "Baseline_Grid": "#636363",
    "BiasNet_ByTime": "#9e9ac8",
    "BiasNet_ByExp": "#6a51a3",
    "FusionNet_ByTime": "#74c476",
    "FusionNet_ByExp": "#238b45",
    "Budgeted_K2": "#31a354",
    "BiasNet+FusionNet_ByExp": "#d95f02",
}

SCENARIO_LABELS = {
    "Baseline_Grid": "Baseline",
    "BiasNet_ByTime": "BiasNet (time)",
    "BiasNet_ByExp": "BiasNet (exp)",
    "FusionNet_ByTime": "FusionNet (time)",
    "FusionNet_ByExp": "FusionNet (exp)",
    "Budgeted_K2": "Budgeted k=2",
    "BiasNet+FusionNet_ByExp": "BiasNet+FusionNet",
}

PRIMARY_SCENARIOS = ["Baseline_Grid", "BiasNet+FusionNet_ByExp", "Budgeted_K2"]


def detect_latest_run(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("*_notebook_cv_eval"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No run directories found under {run_root} matching '*_notebook_cv_eval'.")
    return candidates[-1]


def ensure_columns(df: pd.DataFrame, required: Iterable[str], *, name: str) -> None:
    missing = set(required) - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in {name}: {sorted(missing)}")


def reorder(df: pd.DataFrame, column: str, order: Iterable[str]) -> pd.DataFrame:
    ordering = {value: idx for idx, value in enumerate(order)}
    df = df[df[column].isin(order)].copy()
    df["_order"] = df[column].map(ordering)
    df.sort_values("_order", inplace=True)
    df.drop(columns="_order", inplace=True)
    return df


def plot_cv_summary(df: pd.DataFrame, output: Path) -> None:
    df = reorder(df, "scenario", SUMMARY_SCENARIO_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5.2))

    bars = ax.bar(
        np.arange(len(df)),
        df["RMSE_3D_Mean"],
        color=[PLOT_PALETTE[s] for s in df["scenario"]],
    )

    for idx, rect in enumerate(bars):
        rmse = rect.get_height()
        scenario = df.iloc[idx]["scenario"]
        nees = df.iloc[idx].get("NEES_Mean", np.nan)
        label = SCENARIO_LABELS.get(scenario, scenario)
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rmse + 0.02,
            f"{rmse:.3f} m",
            ha="center",
            va="bottom",
            fontsize=9,
        )
        if not np.isnan(nees):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rmse + 0.14,
                f"NEES {nees:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                color="#444444",
            )

    best_idx = df["RMSE_3D_Mean"].idxmin()
    best_x = np.arange(len(df))[df.index.get_loc(best_idx)]
    best_y = df.loc[best_idx, "RMSE_3D_Mean"]
    ax.scatter(best_x, best_y, s=150, facecolors="none", edgecolors="#ff7f00", linewidths=2)
    ax.annotate(
        "Best overall",
        xy=(best_x, best_y),
        xytext=(best_x + 0.4, best_y + 0.3),
        arrowprops=dict(arrowstyle="->", color="#ff7f00"),
        fontsize=10,
    )

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in df["scenario"]], rotation=25, ha="right")
    ax.set_ylabel("RMSE$_{3D}$ (m)")
    ax.set_title("Cross-validation scenario summary: lower RMSE is better")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output)


def plot_cv_per_fold_deltas(agg: pd.DataFrame, output: Path) -> None:
    baseline = agg[agg["scenario"] == "Baseline_Grid"].set_index(["fold", "exp"])

    deltas = []
    for scenario in SUMMARY_SCENARIO_ORDER:
        if scenario == "Baseline_Grid":
            continue
        current = agg[agg["scenario"] == scenario].set_index(["fold", "exp"])
        common_idx = baseline.index.intersection(current.index)
        if common_idx.empty:
            continue
        diff = baseline.loc[common_idx]["rmse_3d"] - current.loc[common_idx]["rmse_3d"]
        for (fold, exp), value in diff.items():
            deltas.append({
                "scenario": scenario,
                "fold": fold,
                "exp": exp,
                "rmse_gain": value,
            })

    delta_df = pd.DataFrame(deltas)
    if delta_df.empty:
        print("[WARN] No deltas computed for CV per-fold plot; skipping")
        return

    delta_df = reorder(delta_df, "scenario", SUMMARY_SCENARIO_ORDER[1:])

    fig, ax = plt.subplots(figsize=(10, 5.2))

    for idx, scenario in enumerate(delta_df["scenario"].unique()):
        scenario_df = delta_df[delta_df["scenario"] == scenario]
        jitter = (np.random.rand(len(scenario_df)) - 0.5) * 0.15
        ax.scatter(
            np.full(len(scenario_df), idx) + jitter,
            scenario_df["rmse_gain"],
            alpha=0.7,
            color=PLOT_PALETTE.get(scenario, "#3182bd"),
            label=SCENARIO_LABELS.get(scenario, scenario) if idx == 0 else "",
        )
        # Add median line
        median = scenario_df["rmse_gain"].median()
        ax.hlines(
            median,
            idx - 0.3,
            idx + 0.3,
            colors=PLOT_PALETTE.get(scenario, "#3182bd"),
            linestyles="-",
            linewidth=2,
        )

    ax.axhline(0, color="#444444", linewidth=1, linestyle="--")
    ax.set_xticks(range(len(delta_df["scenario"].unique())))
    ax.set_xticklabels([
        SCENARIO_LABELS.get(s, s) for s in delta_df["scenario"].unique()
    ], rotation=25, ha="right")
    ax.set_ylabel("RMSE gain over baseline (m)")
    ax.set_title("Cross-validation per-fold gains versus baseline")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output)


def plot_zigzag_per_experiment(df: pd.DataFrame, output: Path) -> None:
    subset = df[df["scenario"].isin(PRIMARY_SCENARIOS)].copy()
    subset["scenario"] = pd.Categorical(subset["scenario"], PRIMARY_SCENARIOS)
    subset.sort_values(["heldout_exp", "scenario"], inplace=True)

    experiments = subset["heldout_exp"].unique()
    x = np.arange(len(experiments))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5.2))

    for idx, scenario in enumerate(PRIMARY_SCENARIOS):
        scenario_df = subset[subset["scenario"] == scenario]
        rmse_vals = scenario_df["rmse_3d"].to_numpy()
        offsets = x + (idx - 1) * width
        bars = ax.bar(
            offsets,
            rmse_vals,
            width=width,
            color=PLOT_PALETTE.get(scenario, None),
            label=SCENARIO_LABELS.get(scenario, scenario),
        )
        for rect in bars:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                height + 0.05,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([exp.replace("default_3_", "") for exp in experiments], rotation=20, ha="right")
    ax.set_ylabel("RMSE$_{3D}$ (m)")
    ax.set_title("Zigzag LOEO per experiment: architecture comparison")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output)


def plot_zigzag_summary(df: pd.DataFrame, output: Path) -> None:
    df = reorder(df, "scenario", SUMMARY_SCENARIO_ORDER)
    fig, ax = plt.subplots(figsize=(10, 5.2))

    bars = ax.bar(
        np.arange(len(df)),
        df["RMSE_3D_Mean"],
        color=[PLOT_PALETTE[s] for s in df["scenario"]],
    )

    for idx, rect in enumerate(bars):
        rmse = rect.get_height()
        n = df.iloc[idx].get("N_Experiments", np.nan)
        label = f"{rmse:.3f} m"
        if not np.isnan(n):
            label += f"\n(n={int(n)})"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            rmse + 0.05,
            label,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    best_idx = df["RMSE_3D_Mean"].idxmin()
    best_x = np.arange(len(df))[df.index.get_loc(best_idx)]
    best_y = df.loc[best_idx, "RMSE_3D_Mean"]
    ax.scatter(best_x, best_y, s=150, facecolors="none", edgecolors="#ff7f00", linewidths=2)
    ax.annotate(
        "Best zigzag performer",
        xy=(best_x, best_y),
        xytext=(best_x + 0.4, best_y + 0.3),
        arrowprops=dict(arrowstyle="->", color="#ff7f00"),
        fontsize=10,
    )

    ax.set_xticks(np.arange(len(df)))
    ax.set_xticklabels([SCENARIO_LABELS.get(s, s) for s in df["scenario"]], rotation=25, ha="right")
    ax.set_ylabel("RMSE$_{3D}$ (m)")
    ax.set_title("Zigzag LOEO scenario summary")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()

    save_figure(fig, output)


def save_figure(fig: plt.Figure, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[OK] Saved {output.name} (+ PDF)")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (defaults to CWD).")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Specific notebook run directory; auto-detected when omitted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory for output figures. Defaults to <run-root>/figures.",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    run_root = args.run_root.resolve() if args.run_root else detect_latest_run(repo_root / "runs")
    output_root = args.output_root.resolve() if args.output_root else (run_root / "figures")

    cv_summary_path = repo_root / "cv_results_summary.csv"
    cv_agg_path = repo_root / "cv_results_aggregated.csv"
    zigzag_results_path = run_root / "zigzag_loeo_results.csv"
    zigzag_summary_path = run_root / "zigzag_loeo_summary.csv"

    cv_summary = pd.read_csv(cv_summary_path)
    ensure_columns(
        cv_summary,
        ["scenario", "RMSE_3D_Mean", "RMSE_3D_Std", "NEES_Mean", "NEES_Std"],
        name="cv_results_summary.csv",
    )

    cv_agg = pd.read_csv(cv_agg_path)
    ensure_columns(
        cv_agg,
        ["fold", "scenario", "exp", "rmse_3d"],
        name="cv_results_aggregated.csv",
    )

    zigzag_results = pd.read_csv(zigzag_results_path)
    ensure_columns(
        zigzag_results,
        ["heldout_exp", "scenario", "rmse_3d"],
        name=str(zigzag_results_path),
    )

    zigzag_summary = pd.read_csv(zigzag_summary_path)
    ensure_columns(
        zigzag_summary,
        ["scenario", "RMSE_3D_Mean"],
        name=str(zigzag_summary_path),
    )

    plot_cv_summary(cv_summary.copy(), output_root / "cv_architecture_gains_summary.png")
    plot_cv_per_fold_deltas(cv_agg.copy(), output_root / "cv_architecture_gains_per_fold.png")
    plot_zigzag_per_experiment(zigzag_results.copy(), output_root / "zigzag_architecture_gains_per_experiment.png")
    plot_zigzag_summary(zigzag_summary.copy(), output_root / "zigzag_architecture_gains_summary.png")

    print("[OK] All architecture gain plots have been generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
