#!/usr/bin/env python3
"""Aggregate thesis_run_all notebook outputs into a single master results table.

This script consolidates per-experiment, per-fold, and summary CSV outputs produced by
thesis_run_all.ipynb into a canonical table that can be tracked alongside the thesis.
It is intentionally opinionated about the expected directory layout of the most recent
notebook run (``runs/*_notebook_cv_eval``) and the top-level CSV exports that the
notebook writes to the project root.

The resulting CSV uses a normalized schema with the following columns:
    category, statistic, exp, fold, scenario, rmse_3d, rmse_x, rmse_y,
    rmse_z, nees, rmse_3d_std, nees_std, notes, source

If new result artefacts are added to thesis_run_all.ipynb, extend this script so the
master table remains comprehensive.
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List

import pandas as pd

BASE_COLUMNS = [
    "category",
    "statistic",
    "exp",
    "fold",
    "scenario",
    "rmse_3d",
    "rmse_x",
    "rmse_y",
    "rmse_z",
    "nees",
    "rmse_3d_std",
    "nees_std",
    "notes",
    "source",
]


SCENARIO_LABELS = {
    "A_baseline_grid": "Baseline_Grid",
    "B1_bn_by_time": "BiasNet_ByTime",
    "B2_bn_by_exp": "BiasNet_ByExp",
    "C1_fn_by_exp": "FusionNet_ByExp",
    "C2_fn_by_time": "FusionNet_ByTime",
    "D_bnfn_by_exp": "BiasNet+FusionNet_ByExp",
    "E_budget_k2": "Budgeted_K2",
}


@dataclass
class TableRow:
    category: str
    statistic: str
    source: Path
    exp: str | None = None
    fold: float | int | None = None
    scenario: str | None = None
    rmse_3d: float | None = None
    rmse_x: float | None = None
    rmse_y: float | None = None
    rmse_z: float | None = None
    nees: float | None = None
    rmse_3d_std: float | None = None
    nees_std: float | None = None
    notes: str | None = None

    def as_dict(self, repo_root: Path) -> dict:
        row = {col: None for col in BASE_COLUMNS}
        row.update(
            {
                "category": self.category,
                "statistic": self.statistic,
                "exp": self.exp,
                "fold": self.fold,
                "scenario": self.scenario,
                "rmse_3d": self.rmse_3d,
                "rmse_x": self.rmse_x,
                "rmse_y": self.rmse_y,
                "rmse_z": self.rmse_z,
                "nees": self.nees,
                "rmse_3d_std": self.rmse_3d_std,
                "nees_std": self.nees_std,
                "notes": self.notes,
                "source": _relpath(self.source, repo_root),
            }
        )
        return row


def _scenario_label_from_key(key: str) -> str:
    return SCENARIO_LABELS.get(key, key)


def _extract_suffix_int(token: str) -> int | None:
    if not token:
        return None
    if "_" not in token:
        return None
    suffix = token.rsplit("_", 1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _relpath(path: Path, repo_root: Path) -> str:
    try:
        return str(path.relative_to(repo_root))
    except ValueError:
        return str(path)


def detect_latest_run(run_root: Path) -> Path:
    candidates = sorted(run_root.glob("*_notebook_cv_eval"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise SystemExit(f"No run directories found under {run_root} matching '*_notebook_cv_eval'.")
    return candidates[-1]


def find_latest_artifact(run_root: Path, relative_path: str, require_dir: bool | None = None) -> Path | None:
    candidates: List[Path] = []
    for run_dir in sorted(run_root.glob("*_notebook_cv_eval")):
        candidate = run_dir / relative_path
        if not candidate.exists():
            continue
        if require_dir is True and not candidate.is_dir():
            continue
        if require_dir is False and candidate.is_dir():
            continue
        candidates.append(candidate)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_notebook_results(repo_root: Path, run_path: Path, run_root_base: Path) -> List[TableRow]:
    rows: List[TableRow] = []

    def append_if_exists(path: Path, builder: Callable[[Path], None]) -> None:
        if path.exists():
            builder(path)
        else:
            print(f"[WARN] Skipping missing artefact: {path}")

    def append_run_artifact(relative_path: str, builder: Callable[[Path], None], expect_dir: bool | None = None) -> None:
        candidate = run_path / relative_path
        if candidate.exists():
            builder(candidate)
            return
        fallback = find_latest_artifact(run_root_base, relative_path, require_dir=expect_dir)
        if fallback is not None:
            print(f"[INFO] Using fallback artefact for {relative_path}: {fallback}")
            builder(fallback)
        else:
            print(f"[WARN] Skipping missing artefact: {candidate}")

    # Adaptive tracker summaries
    append_run_artifact("zig_sections/adaptive_budget_k1/adaptive_budget_summary.csv", lambda p: _build_adaptive(rows, p))

    # Fixed tracker summaries (ifo001)
    append_run_artifact("zig_sections/fixed_tracker_ifo001/fixed_tracker_summary.csv", lambda p: _build_fixed(rows, p, tracker_id="ifo001"))

    # Additional fixed tracker experiments (ifo002, etc.)
    append_run_artifact("fixed_tracker_ifo002/all_summary.csv", lambda p: _build_fixed(rows, p, tracker_id="ifo002"))

    # Default experiment single-tracker replays
    append_run_artifact("default_single_tracker/default_single_tracker_summary.csv", lambda p: _build_fixed(rows, p, tracker_id="ifo001"))
    append_run_artifact("default_single_tracker_random/default_single_tracker_random_summary.csv", lambda p: _build_fixed(rows, p, tracker_id="ifo001"))

    # Tracker strategy comparisons (baseline vs adaptive vs fixed)
    append_run_artifact("figures/tracker_strategy_comparison.csv", lambda p: _build_tracker_comparisons(rows, p))

    # Zigzag LOEO per-fold results and summaries
    append_run_artifact("zigzag_loeo_results.csv", lambda p: _build_zigzag_results(rows, p))
    append_run_artifact("zigzag_loeo_summary.csv", lambda p: _build_zigzag_summary(rows, p))

    # Pattern-based cross-validation (held-out trajectory patterns)
    append_run_artifact("all_summaries.csv", lambda p: _build_pattern_cv(rows, p))

    # NonZig -> Zig generalization folds
    append_run_artifact("cv_nonzig", lambda p: _build_nonzig_generalization(rows, p), expect_dir=True)

    # Cross-validation artefacts (written to project root)
    append_if_exists(repo_root / "cv_results_aggregated.csv", lambda p: _build_cv_per_fold(rows, p))
    append_if_exists(repo_root / "cv_results_summary.csv", lambda p: _build_cv_summary(rows, p))
    append_if_exists(repo_root / "cv_results_comprehensive_k1_k2.csv", lambda p: _build_cv_comprehensive(rows, p))
    append_if_exists(repo_root / "baseline_vs_budgeted_k2_comparison.csv", lambda p: _build_cv_deltas(rows, p))

    return rows


def _build_adaptive(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append(
            TableRow(
                category="Adaptive tracker",
                statistic="per_experiment",
                source=path,
                exp=str(r.get("exp")),
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("rmse_3d")),
                rmse_x=_to_float(r.get("rmse_x")),
                rmse_y=_to_float(r.get("rmse_y")),
                rmse_z=_to_float(r.get("rmse_z")),
                nees=_to_float(r.get("nees")),
            )
        )


def _build_fixed(rows: List[TableRow], path: Path, tracker_id: str) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        note_parts = []
        if "fixed_tracker" in df.columns:
            note_parts.append(f"fixed_tracker={r.get('fixed_tracker')}")
        else:
            note_parts.append(f"fixed_tracker={tracker_id}")
        rows.append(
            TableRow(
                category="Fixed tracker",
                statistic="per_experiment",
                source=path,
                exp=str(r.get("exp")),
                scenario=str(r.get("scenario", f"Fixed_{tracker_id}")),
                rmse_3d=_to_float(r.get("rmse_3d")),
                rmse_x=_to_float(r.get("rmse_x")),
                rmse_y=_to_float(r.get("rmse_y")),
                rmse_z=_to_float(r.get("rmse_z")),
                nees=_to_float(r.get("nees")),
                notes=", ".join(note_parts),
            )
        )


def _build_tracker_comparisons(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        exp = str(r.get("exp_base"))
        rows.extend(
            [
                TableRow(
                    category="Tracker comparison",
                    statistic="per_experiment",
                    source=path,
                    exp=exp,
                    scenario="Baseline",
                    rmse_3d=_to_float(r.get("rmse_3d_baseline")),
                    nees=_to_float(r.get("nees_baseline")),
                ),
                TableRow(
                    category="Tracker comparison",
                    statistic="per_experiment",
                    source=path,
                    exp=exp,
                    scenario="Adaptive",
                    rmse_3d=_to_float(r.get("rmse_3d_adaptive")),
                    nees=_to_float(r.get("nees_adaptive")),
                ),
                TableRow(
                    category="Tracker comparison",
                    statistic="per_experiment",
                    source=path,
                    exp=exp,
                    scenario="Fixed_ifo001",
                    rmse_3d=_to_float(r.get("rmse_3d_fixed")),
                    nees=_to_float(r.get("nees_fixed")),
                ),
                TableRow(
                    category="Tracker comparison",
                    statistic="delta",
                    source=path,
                    exp=exp,
                    scenario="Adaptive - Baseline",
                    rmse_3d=_to_float(r.get("adaptive_minus_baseline")),
                ),
                TableRow(
                    category="Tracker comparison",
                    statistic="delta",
                    source=path,
                    exp=exp,
                    scenario="Fixed - Adaptive",
                    rmse_3d=_to_float(r.get("fixed_minus_adaptive")),
                ),
                TableRow(
                    category="Tracker comparison",
                    statistic="delta",
                    source=path,
                    exp=exp,
                    scenario="Fixed - Baseline",
                    rmse_3d=_to_float(r.get("fixed_minus_baseline")),
                ),
            ]
        )


def _build_zigzag_results(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append(
            TableRow(
                category="Zigzag LOEO",
                statistic="per_fold",
                source=path,
                exp=str(r.get("heldout_exp")),
                fold=_to_float(r.get("fold")),
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("rmse_3d")),
                rmse_x=_to_float(r.get("rmse_x")),
                rmse_y=_to_float(r.get("rmse_y")),
                rmse_z=_to_float(r.get("rmse_z")),
                nees=_to_float(r.get("nees")),
            )
        )


def _build_zigzag_summary(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        notes = f"experiments={r.get('N_Experiments')}"
        rows.append(
            TableRow(
                category="Zigzag LOEO",
                statistic="scenario_summary",
                source=path,
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("RMSE_3D_Mean")),
                rmse_3d_std=_to_float(r.get("RMSE_3D_Std")),
                nees=_to_float(r.get("NEES_Mean")),
                nees_std=_to_float(r.get("NEES_Std")),
                notes=notes,
            )
        )


def _build_pattern_cv(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    if "exp_dir" not in df.columns:
        print(f"[WARN] Pattern CV summary missing 'exp_dir' column: {path}")
        return

    summary_records = []
    for _, r in df.iterrows():
        exp_dir = str(r.get("exp_dir"))
        parts = exp_dir.split("/")
        if len(parts) < 4:
            print(f"[WARN] Skipping malformed pattern entry '{exp_dir}' in {path}")
            continue

        fold_token = parts[1]
        scenario_key = parts[2]
        exp_name = parts[-1]
        fold_number = _extract_suffix_int(fold_token)
        if fold_number is None:
            print(f"[WARN] Could not parse fold from '{fold_token}' in {path}")
            continue

        rmse_x = _to_float(r.get("rmse_x"))
        rmse_y = _to_float(r.get("rmse_y"))
        rmse_z = _to_float(r.get("rmse_z"))
        rmse_3d = _to_float(r.get("rmse_3d"))
        nees = _to_float(r.get("nees"))

        scenario_label = _scenario_label_from_key(scenario_key)
        rows.append(
            TableRow(
                category="Pattern CV",
                statistic="per_fold",
                source=path,
                exp=exp_name,
                fold=fold_number,
                scenario=scenario_label,
                rmse_3d=rmse_3d,
                rmse_x=rmse_x,
                rmse_y=rmse_y,
                rmse_z=rmse_z,
                nees=nees,
                notes=f"scenario_key={scenario_key}",
            )
        )

        summary_records.append(
            {
                "scenario": scenario_label,
                "scenario_key": scenario_key,
                "fold": fold_number,
                "rmse_x": rmse_x,
                "rmse_y": rmse_y,
                "rmse_z": rmse_z,
                "rmse_3d": rmse_3d,
                "nees": nees,
            }
        )

    if not summary_records:
        return

    summary_df = pd.DataFrame(summary_records)
    grouped = (
        summary_df.groupby(["scenario", "scenario_key"])
        .agg(
            rmse_x=("rmse_x", "mean"),
            rmse_y=("rmse_y", "mean"),
            rmse_z=("rmse_z", "mean"),
            rmse_3d=("rmse_3d", "mean"),
            nees=("nees", "mean"),
            rmse_3d_std=("rmse_3d", "std"),
            nees_std=("nees", "std"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.fillna({"rmse_3d_std": 0.0, "nees_std": 0.0})

    for _, agg in grouped.iterrows():
        notes = f"scenario_key={agg['scenario_key']}, folds={int(agg['fold_count'])}"
        rows.append(
            TableRow(
                category="Pattern CV",
                statistic="scenario_summary",
                source=path,
                scenario=str(agg["scenario"]),
                rmse_x=_to_float(agg.get("rmse_x")),
                rmse_y=_to_float(agg.get("rmse_y")),
                rmse_z=_to_float(agg.get("rmse_z")),
                rmse_3d=_to_float(agg.get("rmse_3d")),
                nees=_to_float(agg.get("nees")),
                rmse_3d_std=_to_float(agg.get("rmse_3d_std")),
                nees_std=_to_float(agg.get("nees_std")),
                notes=notes,
            )
        )


def _build_nonzig_generalization(rows: List[TableRow], base_dir: Path) -> None:
    if not base_dir.is_dir():
        print(f"[WARN] NonZig generalization directory missing: {base_dir}")
        return

    summary_records = []
    for fold_dir in sorted(base_dir.glob("cv_fold_*")):
        if not fold_dir.is_dir():
            continue
        fold_number = _extract_suffix_int(fold_dir.name)
        if fold_number is None:
            print(f"[WARN] Could not parse fold from '{fold_dir.name}' in {base_dir}")
            continue
        for scenario_dir in sorted(fold_dir.iterdir()):
            if not scenario_dir.is_dir():
                continue
            scenario_key = scenario_dir.name
            summary_path = scenario_dir / "all_summary.csv"
            if not summary_path.exists():
                continue

            df = pd.read_csv(summary_path)
            for _, r in df.iterrows():
                exp_name = str(r.get("exp"))
                rmse_x = _to_float(r.get("rmse_x"))
                rmse_y = _to_float(r.get("rmse_y"))
                rmse_z = _to_float(r.get("rmse_z"))
                rmse_3d = _to_float(r.get("rmse_3d"))
                nees = _to_float(r.get("nees"))

                scenario_label = _scenario_label_from_key(scenario_key)
                rows.append(
                    TableRow(
                        category="NonZig->Zig generalization",
                        statistic="per_fold",
                        source=summary_path,
                        exp=exp_name,
                        fold=fold_number,
                        scenario=scenario_label,
                        rmse_x=rmse_x,
                        rmse_y=rmse_y,
                        rmse_z=rmse_z,
                        rmse_3d=rmse_3d,
                        nees=nees,
                        notes=f"scenario_key={scenario_key}",
                    )
                )

                summary_records.append(
                    {
                        "scenario": scenario_label,
                        "scenario_key": scenario_key,
                        "fold": fold_number,
                        "rmse_x": rmse_x,
                        "rmse_y": rmse_y,
                        "rmse_z": rmse_z,
                        "rmse_3d": rmse_3d,
                        "nees": nees,
                    }
                )

    if not summary_records:
        return

    summary_df = pd.DataFrame(summary_records)
    grouped = (
        summary_df.groupby(["scenario", "scenario_key"])
        .agg(
            rmse_x=("rmse_x", "mean"),
            rmse_y=("rmse_y", "mean"),
            rmse_z=("rmse_z", "mean"),
            rmse_3d=("rmse_3d", "mean"),
            nees=("nees", "mean"),
            rmse_3d_std=("rmse_3d", "std"),
            nees_std=("nees", "std"),
            fold_count=("fold", "nunique"),
        )
        .reset_index()
    )
    grouped = grouped.fillna({"rmse_3d_std": 0.0, "nees_std": 0.0})

    for _, agg in grouped.iterrows():
        notes = f"scenario_key={agg['scenario_key']}, folds={int(agg['fold_count'])}"
        rows.append(
            TableRow(
                category="NonZig->Zig generalization",
                statistic="scenario_summary",
                source=base_dir,
                scenario=str(agg["scenario"]),
                rmse_x=_to_float(agg.get("rmse_x")),
                rmse_y=_to_float(agg.get("rmse_y")),
                rmse_z=_to_float(agg.get("rmse_z")),
                rmse_3d=_to_float(agg.get("rmse_3d")),
                nees=_to_float(agg.get("nees")),
                rmse_3d_std=_to_float(agg.get("rmse_3d_std")),
                nees_std=_to_float(agg.get("nees_std")),
                notes=notes,
            )
        )


def _build_cv_per_fold(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        rows.append(
            TableRow(
                category="Cross-validation per-fold",
                statistic="per_fold",
                source=path,
                exp=str(r.get("exp")),
                fold=_to_float(r.get("fold")),
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("rmse_3d")),
                rmse_x=_to_float(r.get("rmse_x")),
                rmse_y=_to_float(r.get("rmse_y")),
                rmse_z=_to_float(r.get("rmse_z")),
                nees=_to_float(r.get("nees")),
            )
        )


def _build_cv_summary(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        notes = []
        for field in ("RMSE_Rank", "NEES_Rank", "N_Folds"):
            if field in r:
                notes.append(f"{field.lower()}={r[field]}")
        rows.append(
            TableRow(
                category="Cross-validation summary",
                statistic="scenario_summary",
                source=path,
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("RMSE_3D_Mean")),
                rmse_3d_std=_to_float(r.get("RMSE_3D_Std")),
                nees=_to_float(r.get("NEES_Mean")),
                nees_std=_to_float(r.get("NEES_Std")),
                notes=", ".join(notes) if notes else None,
            )
        )


def _build_cv_comprehensive(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        notes = f"Method={r.get('Method')}, Budget_K={r.get('Budget_K')}, folds={r.get('N_Folds')}"
        rows.append(
            TableRow(
                category="Cross-validation comprehensive",
                statistic="per_model",
                source=path,
                scenario=str(r.get("scenario")),
                rmse_3d=_to_float(r.get("RMSE_3D_Mean")),
                rmse_3d_std=_to_float(r.get("RMSE_3D_Std")),
                nees=_to_float(r.get("NEES_Mean")),
                nees_std=_to_float(r.get("NEES_Std")),
                notes=notes,
            )
        )


def _build_cv_deltas(rows: List[TableRow], path: Path) -> None:
    df = pd.read_csv(path)
    for _, r in df.iterrows():
        note = (
            f"baseline={_to_float(r.get('baseline_grid'))}, "
            f"budgeted={_to_float(r.get('budgeted_k2'))}, "
            f"diff_percent={_to_float(r.get('diff_percent'))}"
        )
        rows.append(
            TableRow(
                category="Cross-validation delta",
                statistic="delta_per_fold",
                source=path,
                exp=str(r.get("experiment")),
                fold=_to_float(r.get("fold")),
                scenario="Budgeted_K2 - Baseline",
                rmse_3d=_to_float(r.get("difference")),
                notes=note,
            )
        )


def _to_float(value) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed):
        return None
    return parsed


def build_dataframe(rows: Iterable[TableRow], repo_root: Path) -> pd.DataFrame:
    df = pd.DataFrame([row.as_dict(repo_root) for row in rows], columns=BASE_COLUMNS)
    df = df.sort_values(["category", "statistic", "scenario", "exp", "fold"], na_position="last")
    df.reset_index(drop=True, inplace=True)
    return df


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Repository root (defaults to CWD).")
    parser.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="Specific notebook run directory. If omitted the latest runs/*_notebook_cv_eval directory is used.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV path. Defaults to <run-root>/figures/notebook_master_results_table.csv.",
    )
    parser.add_argument(
        "--also-root-copy",
        action="store_true",
        help="Additionally write notebook_master_results_table.csv to the repository root.",
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    repo_root: Path = args.repo_root.resolve()
    run_root_base = repo_root / "runs"

    if args.run_root is None:
        run_path = detect_latest_run(run_root_base)
        print(f"[INFO] Auto-detected latest notebook run: {run_path}")
    else:
        run_path = args.run_root.resolve()
        if not run_path.exists():
            raise SystemExit(f"Specified run directory does not exist: {run_path}")

    rows = load_notebook_results(repo_root, run_path, run_root_base)
    if not rows:
        raise SystemExit("No result rows collected. Check that notebook outputs exist.")

    df = build_dataframe(rows, repo_root)

    output_path = args.output
    if output_path is None:
        output_path = run_path / "figures" / "notebook_master_results_table.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[OK] Wrote master results table: {output_path}")

    if args.also_root_copy:
        root_copy = repo_root / "notebook_master_results_table.csv"
        df.to_csv(root_copy, index=False)
        print(f"[OK] Copied master results table to: {root_copy}")

    print("\nRow counts by category:")
    print(df.groupby(["category", "statistic"]).size())

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
