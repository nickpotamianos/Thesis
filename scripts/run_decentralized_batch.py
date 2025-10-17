#!/usr/bin/env python3
"""
run_decentralized_batch.py
-------------------------
Execute the full decentralized baseline and ML evaluation across the seven
pattern folds used in the thesis experiments. Results are written to
`outputs_swarm/` so downstream plotting scripts can consume them directly.

This script mirrors the logic from `thesis_run_all.ipynb` Step 3 without the
notebook dependency.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PYTHON_EXE = ROOT / "miluv_env" / "bin" / "python"

if not PYTHON_EXE.exists():  # pragma: no cover - sanity guard
    raise FileNotFoundError(
        "miluv_env python executable not found. Ensure the virtual environment exists."
    )

MODEL_BASE = ROOT / "runs" / "20250910_160447_notebook_cv_eval" / "models"
if not MODEL_BASE.exists():  # pragma: no cover
    raise FileNotFoundError(
        "Required model artifacts not found. Expected directory:"
        f" {MODEL_BASE}"
    )

FOLD_EXPERIMENTS: Dict[int, str] = {
    1: "default_3_movingTriangle_0b",
    2: "default_3_random2_0",
    3: "default_3_random3_0b",
    4: "default_3_random3_1",
    5: "default_3_random3_2",
    6: "default_3_random_0",
    7: "default_3_random_0b",
}

OUTPUT_MAP = {
    "baseline": ROOT / "outputs_swarm" / "baseline_0b_corrected",
    "ml": ROOT / "outputs_swarm" / "ml_0b_corrected",
}

COMMON_NODE_FLAGS: List[str] = [
    "--use_height_tf",
    "--uwb_std",
    "0.8",
    "--pair_corr",
    "0.3",
    "--sigma_a_xy",
    "3.0",
    "--sigma_a_z",
    "1.5",
    "--los_influence",
    "0",
    "--geom_influence",
    "0",
    "--online_tune",
    "--gate_target",
    "0.90",
    "--q_adapt",
    "--control_mode",
    "none",
]

LOGGER_BASE_CMD = [
    str(PYTHON_EXE),
    "-m",
    "agents.logger",
    "--target",
    "ifo003",
    "--method",
    "gossip",
    "--rounds",
    "3",
    "--ci_objective",
    "trace",
    "--fanout",
    "--planner",
    "none",
]

NODE_BASE_CMD = [
    str(PYTHON_EXE),
    "-m",
    "agents.robot_node",
    "--target",
    "ifo003",
]


def clean_processes() -> None:
    """Terminate lingering agent processes."""
    subprocess.run(["pkill", "-f", "agents"], capture_output=True)
    time.sleep(1)


def get_model_paths(fold: int) -> tuple[Optional[str], Optional[str]]:
    fusion_dir = MODEL_BASE / f"pattern_fold_{fold}_fn" / "fusionnet_by_exp"
    bias_dir = MODEL_BASE / f"pattern_fold_{fold}_bn_time_byexp"
    if fold == 1:
        bias_dir = MODEL_BASE / "pattern_fold_1_bn_time_v2_byexp"
    return (
        str(fusion_dir) if fusion_dir.exists() else None,
        str(bias_dir) if bias_dir.exists() else None,
    )


def run_experiment(experiment: str, fold: int, variant: str, timeout_minutes: int = 20) -> Optional[Dict[str, object]]:
    """Run a single decentralized experiment and return summary stats."""

    out_root = OUTPUT_MAP[variant]
    exp_out = out_root / f"{experiment}_fold{fold}_ifo003"
    result_dir = exp_out / f"{experiment}_ifo003"

    exp_out.mkdir(parents=True, exist_ok=True)
    if result_dir.exists():
        shutil.rmtree(result_dir)

    clean_processes()

    logger_cmd = LOGGER_BASE_CMD + ["--exp", experiment, "--out", str(exp_out)]
    node_params = COMMON_NODE_FLAGS.copy()

    if variant == "ml":
        fusion_dir, bias_dir = get_model_paths(fold)
        if fusion_dir is None or bias_dir is None:
            print(f"  ✗ Missing models for fold {fold}; skipping ML variant")
            return None
        logger_cmd.extend(["--fusionnet_dir", fusion_dir])
        node_params.extend(["--biasnet_dir", bias_dir, "--bias_gain", "0.6"])

    logger_proc = subprocess.Popen(logger_cmd, cwd=str(ROOT))
    time.sleep(2)

    node_procs = []
    for robot_id in ["ifo001", "ifo002"]:
        cmd = NODE_BASE_CMD + ["--id", robot_id, "--exp", experiment] + node_params
        proc = subprocess.Popen(cmd, cwd=str(ROOT))
        node_procs.append(proc)
        time.sleep(0.5)

    max_wait = timeout_minutes * 60
    start = time.time()
    while logger_proc.poll() is None and (time.time() - start) < max_wait:
        time.sleep(2)

    # Clean up
    for proc in [logger_proc, *node_procs]:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    clean_processes()

    summary_file = result_dir / "summary.csv"
    if not summary_file.exists():
        print(f"  ✗ No results for {variant} {experiment}")
        return None

    try:
        df = pd.read_csv(summary_file)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ✗ Failed reading summary for {experiment}: {exc}")
        return None

    if df.empty:
        print(f"  ✗ Empty summary for {variant} {experiment}")
        return None

    rec = df.iloc[0].to_dict()
    return {
        "experiment": experiment,
        "fold": fold,
        "variant": variant,
        "rmse_3d": float(rec.get("rmse_3d", float("nan"))),
        "nees": float(rec.get("nees", float("nan"))),
        "summary_path": str(summary_file),
    }


def main() -> None:
    results: List[Dict[str, object]] = []
    print("=== Running Decentralized UDP batch ===")
    for fold, experiment in FOLD_EXPERIMENTS.items():
        print(f"\n--- Fold {fold}: {experiment} ---")

        baseline = run_experiment(experiment, fold, "baseline")
        if baseline:
            results.append(baseline)
            print(
                f"  ✓ Baseline: RMSE {baseline['rmse_3d']:.3f}m | NEES {baseline['nees']:.3f}"
            )

        ml = run_experiment(experiment, fold, "ml")
        if ml:
            results.append(ml)
            print(f"  ✓ ML: RMSE {ml['rmse_3d']:.3f}m | NEES {ml['nees']:.3f}")

    if not results:
        print("No decentralized results produced.")
        return

    out_csv = ROOT / "outputs_swarm" / "udp_batch_summary.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["experiment", "fold", "variant", "rmse_3d", "nees", "summary_path"]
        )
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"\n=== Completed {len(results)} runs ===")
    print(f"Summary table -> {out_csv}")
if __name__ == "__main__":
    main()
