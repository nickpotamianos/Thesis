#!/usr/bin/env python3
"""
thesis_run_all.py
=================
A single, thesis-grade orchestrator that:
  1) Discovers experiments (non-zigzag vs zigzag)
  2) Collects BiasNet + FusionNet data once per (exp,target) [collect-only]
  3) Trains BiasNet and FusionNet (by_exp and by_time splits) on aggregated TRAIN sets
  4) Evaluates on held-out TEST experiments with scenarios:
        - Baseline (CI grid)
        - BN only (learned bias)
        - FN only (learned CI weights)
        - BN+FN (matched splits)
        - Budgeted team size (top-k)
        - Heuristic planner, EIG planner
        - Decentralized Gossip CI (offline simulation)
  5) Runs **UDP decentralized** demos: grid, gossip, learned (FusionNet)
  6) Builds master tables with RMSE/NEES and saves per-run artifacts

STRICT SETTINGS (as requested; applied everywhere):
  --use_height
  --use_height_tf
  --uwb_std 0.8
  --pair_corr 0.3
  --sigma_a_xy 3.0
  --sigma_a_z  1.5
  --ci_method grid
  --ci_objective trace
  --los_influence 0
  --geom_influence 0
  --ema_alpha 0.0
  --online_tune
  --online_r_min_scale 0.75
  --online_r_max_scale 3.0
  --gate_target 0.90
  --gate_sigma_init 4.0
  --q_adapt

NOTES
-----
- LOS is entirely disabled: we NEVER pass --use_los and set influences to 0.
- Outputs never overwrite: every run lives in runs/<timestamp>_<tag>/...
- Collection uses --collect_only where appropriate; evaluation never re-collects.
- UDP decentralized tests are executed on separate multicast ports to avoid clashes.
"""

from __future__ import annotations
import argparse
import datetime as _dt
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ---------- utility ----------

def stamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def run(cmd: List[str], cwd: Path | None = None) -> None:
    print("[RUN]", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=(str(cwd) if cwd else None))

def spawn(cmd: List[str], cwd: Path | None = None) -> subprocess.Popen:
    print("[SPAWN]", " ".join(cmd), flush=True)
    return subprocess.Popen(cmd, cwd=(str(cwd) if cwd else None))

def unique_dir(base: Path) -> Path:
    """Return a non-existent directory path by adding _v2, _v3, ... if needed."""
    if not base.exists():
        return base
    i = 2
    while True:
        cand = base.with_name(f"{base.name}_v{i}")
        if not cand.exists():
            return cand
        i += 1

def discover_experiments(expdir: Path) -> Tuple[List[str], List[str], List[str]]:
    """
    Return (ALL, NONZIG, ZIG) directories in expdir (filter *cir*, *obstacle*, *noapriltag*, *onetag*).
    """
    if not expdir.is_dir():
        raise FileNotFoundError(f"Experiment root not found: {expdir}")
    all_exps = []
    for p in sorted(expdir.iterdir()):
        if not p.is_dir():
            continue
        low = p.name.lower()
        if any(k in low for k in ("cir", "obstacle", "noapriltag", "onetag")):
            continue
        all_exps.append(p.name)
    zig = [e for e in all_exps if "zigzag" in e.lower()]
    base = [e for e in all_exps if e not in zig]
    print(f"[DATASET] exps={len(all_exps)}  nonzig={len(base)}  zig={len(zig)}")
    return all_exps, base, zig

# ---------- fixed parameters you asked for ----------

def best_flags() -> List[str]:
    return [
        "--use_height",
        "--use_height_tf",
        "--uwb_std", "0.8",
        "--pair_corr", "0.3",
        "--sigma_a_xy", "3.0",
        "--sigma_a_z",  "1.5",
        "--ci_method", "grid",
        "--ci_objective", "trace",
        "--los_influence", "0",
        "--geom_influence", "0",
        "--ema_alpha", "0.0",
        "--online_tune",
        "--online_r_min_scale", "0.75",
        "--online_r_max_scale", "3.0",
        "--gate_target", "0.90",
        "--gate_sigma_init", "4.0",
        "--q_adapt",
        "--r_floor_blend", "0.5",
    ]

# ---------- data collection ----------

def ensure_collect(exp: str, target: str, outdir: Path, root: Path) -> None:
    """
    Run a collect-only pass ONCE per (exp,target) if artifacts are missing.
    It will NOT overwrite; it writes under a unique subdir if needed.
    """
    outdir = unique_dir(outdir)
    outdir.mkdir(parents=True, exist_ok=False)
    print(f"[COLLECT] {exp}:{target} -> {outdir}")
    cmd = [
        sys.executable, "swarm_target_tracking.py",
        "--exp", exp, "--target", target,
        *best_flags(),
        "--collect_bias", "--collect_fusion", "--collect_only",
        "--out", str(outdir)
    ]
    run(cmd, cwd=root)

def build_dataset(cache_root: Path, out_dir: Path, targets: List[str], exps: List[str]) -> None:
    """
    Merge cached bias_samples.jsonl and fusion_snaps.jsonl across (targets × exps)
    into one dataset folder (fresh; never overwrite).
    """
    out_dir = unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    bias_path = out_dir / "bias_samples.jsonl"
    fuse_path = out_dir / "fusion_snaps.jsonl"

    def append_jsonl(src: Path, dst: Path) -> int:
        n = 0
        if not src.exists():
            return 0
        mode = "at" if dst.exists() else "wt"
        with open(src, "r") as fin, open(dst, mode) as fout:
            for line in fin:
                s = line.strip()
                if s:
                    fout.write(s + "\n")
                    n += 1
        return n

    tot_b = tot_f = 0
    for tgt in targets:
        for exp in exps:
            # The per-collect subdir is unknown—find the newest one under cache_root/tgt matching exp*
            base = cache_root / tgt
            if not base.exists():
                continue
            cand_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith(exp)],
                               key=lambda p: p.stat().st_mtime)
            if not cand_dirs:
                continue
            d = cand_dirs[-1]  # pick newest
            b = d / "bias_samples.jsonl"
            f = d / "fusion_snaps.jsonl"
            if b.exists():
                tot_b += append_jsonl(b, bias_path)
            if f.exists():
                tot_f += append_jsonl(f, fuse_path)
            gz = d / "fusion_snaps.jsonl.gz"
            if (not f.exists()) and gz.exists():
                import gzip
                with gzip.open(gz, "rt") as fin, open(fuse_path, "at" if fuse_path.exists() else "wt") as fout:
                    for line in fin:
                        s = line.strip()
                        if s:
                            fout.write(s + "\n"); tot_f += 1

    print(f"[DATASET] -> {out_dir}")
    print(f"  bias_samples.jsonl : {tot_b} lines")
    print(f"  fusion_snaps.jsonl : {tot_f} lines")
    if tot_b == 0 or tot_f == 0:
        print("[WARN] Empty dataset—check filters/collections.")

# ---------- training ----------

def train_biasnet(ds_dir: Path, out_dir: Path, root: Path) -> Path:
    out_dir = unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cmd = [
        sys.executable, "-m", "swarm_ml.train_biasnet_cli",
        "--samples", str(ds_dir / "bias_samples.jsonl"),
        "--out", str(out_dir),
        "--split_mode", "by_time", "--epochs", "30", "--seed", "0"
    ]
    run(cmd, cwd=root)
    # also a by_exp variant for matched BN+FN
    out_dir_exp = unique_dir(out_dir.parent / (out_dir.name + "_byexp"))
    run([
        sys.executable, "-m", "swarm_ml.train_biasnet_cli",
        "--samples", str(ds_dir / "bias_samples.jsonl"),
        "--out", str(out_dir_exp),
        "--split_mode", "by_exp", "--epochs", "30", "--seed", "0"
    ], cwd=root)
    return out_dir  # return the by_time primary; caller can infer _byexp sibling

def train_fusionnet(ds_dir: Path, out_dir: Path, root: Path) -> Path:
    out_dir = unique_dir(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    # by_exp
    out_exp = out_dir / "fusionnet_by_exp"
    run([
        sys.executable, "-m", "swarm_ml.train_fusionnet_cli",
        "--snaps", str(ds_dir / "fusion_snaps.jsonl"),
        "--out", str(out_exp),
        "--split_mode", "by_exp", "--epochs", "20", "--seed", "0"
    ], cwd=root)
    # by_time
    out_time = out_dir / "fusionnet_by_time"
    run([
        sys.executable, "-m", "swarm_ml.train_fusionnet_cli",
        "--snaps", str(ds_dir / "fusion_snaps.jsonl"),
        "--out", str(out_time),
        "--split_mode", "by_time", "--epochs", "20", "--seed", "0"
    ], cwd=root)
    return out_dir

# ---------- evaluation ----------

def eval_scenarios(exp: str,
                   target: str,
                   model_dirs: Dict[str, Path],
                   out_root: Path,
                   root: Path,
                   budget_k: int | None = 2,
                   run_planners: bool = True,
                   decentralized: bool = True) -> None:
    """
    Evaluate several scenarios on a single test experiment.
    model_dirs: {"bn_time":..., "bn_exp":..., "fn_root":...}
    """
    out_root = unique_dir(out_root); out_root.mkdir(parents=True, exist_ok=False)
    common = best_flags()

    def run_eval(tag: str, extra: List[str]) -> None:
        d = out_root / tag
        d.mkdir(parents=True, exist_ok=False)
        cmd = [sys.executable, "swarm_target_tracking.py",
               "--exp", exp, "--target", target, *common,
               "--out", str(d)] + extra
        run(cmd, cwd=root)
        # summary table (flat) for this scenario
        run([sys.executable, "swarm_eval_table.py", "--root", str(d)], cwd=root)

    # A) Baseline centralized CI grid
    run_eval("A_baseline_grid", [])

    # B) BN only (by_time & by_exp)
    if "bn_time" in model_dirs:
        run_eval("B1_bn_by_time", ["--biasnet_dir", str(model_dirs["bn_time"]), "--bias_gain", "0.6"])
    if "bn_exp" in model_dirs:
        run_eval("B2_bn_by_exp",  ["--biasnet_dir", str(model_dirs["bn_exp"]),  "--bias_gain", "0.6"])

    # C) FN only (learned CI weights)
    if "fn_root" in model_dirs:
        fn_time = model_dirs["fn_root"] / "fusionnet_by_time"
        fn_exp  = model_dirs["fn_root"] / "fusionnet_by_exp"
        if fn_exp.exists():
            run_eval("C1_fn_by_exp",  ["--ci_method", "learned", "--fusionnet_dir", str(fn_exp)])
        if fn_time.exists():
            run_eval("C2_fn_by_time", ["--ci_method", "learned", "--fusionnet_dir", str(fn_time)])

    # D) BN + FN (matched splits)
    if "bn_exp" in model_dirs and "fn_root" in model_dirs:
        run_eval("D_bnfn_by_exp",  ["--ci_method", "learned",
                                    "--biasnet_dir", str(model_dirs["bn_exp"]),  "--bias_gain", "0.6",
                                    "--fusionnet_dir", str(model_dirs["fn_root"] / "fusionnet_by_exp")])
    if "bn_time" in model_dirs and "fn_root" in model_dirs:
        run_eval("D_bnfn_by_time", ["--ci_method", "learned",
                                    "--biasnet_dir", str(model_dirs["bn_time"]), "--bias_gain", "0.6",
                                    "--fusionnet_dir", str(model_dirs["fn_root"] / "fusionnet_by_time")])

    # E) Budgeted: top-k trackers under learned weights (by_time)
    if budget_k and "fn_root" in model_dirs:
        run_eval(f"E_budget_k{budget_k}", ["--ci_method", "learned",
                                           "--fusionnet_dir", str(model_dirs["fn_root"] / "fusionnet_by_time"),
                                           "--budget_k", str(budget_k)])

    # F) Active sensing planners (heuristic + EIG) with BN+Tuner grid-CI
    if run_planners and "bn_time" in model_dirs:
        run_eval("F_planner_heuristic", ["--biasnet_dir", str(model_dirs["bn_time"]), "--bias_gain", "0.6",
                                         "--planner", "heuristic"])
        run_eval("F_planner_eig",       ["--biasnet_dir", str(model_dirs["bn_time"]), "--bias_gain", "0.6",
                                         "--planner", "eig"])

    # G) Offline decentralized Gossip CI (simulated)
    if decentralized:
        run_eval("G_gossip_ci", ["--decentralized", "--comm_p", "0.7", "--comm_drop", "0.2", "--comm_rounds", "3"])

# ---------- UDP decentralized demos ----------

def udp_port(i: int) -> int:
    return 5001 + i

def udp_run(exp: str,
            target: str,
            trackers: List[str],
            bn_dir: Path | None,
            fn_dir_time: Path | None,
            out_root: Path,
            root: Path) -> None:
    """
    Launch UDP decentralized demo runs for methods:
      - grid (centralized CI at logger, decentralized nodes via UDP)
      - gossip (GossipFuser at logger)
      - learned (FusionNet at logger)
    Each in its own multicast port to avoid collisions.
    """
    out_root = unique_dir(out_root); out_root.mkdir(parents=True, exist_ok=False)
    methods = [
        ("grid",   None),
        ("gossip", None),
        ("learned", fn_dir_time),
    ]
    # Fixed node flags, strictly per your request (no LOS)
    node_flags = [
        "--uwb_std", "0.8",
        "--pair_corr", "0.3",
        "--sigma_a_xy", "3.0",
        "--sigma_a_z", "1.5",
        "--use_height_tf",
        "--los_influence", "0",
        "--geom_influence", "0",
        "--online_tune",
        "--online_r_min_scale", "0.75",
        "--online_r_max_scale", "3.0",
        "--gate_target", "0.90",
        "--gate_sigma_init", "4.0",
        "--q_adapt",
        "--r_floor_blend", "0.5",
        "--control_mode", "sim", "--control_rate", "5",
    ]
    if bn_dir is not None:
        node_flags += ["--biasnet_dir", str(bn_dir), "--bias_gain", "0.6"]

    for m_ix, (method, fn_dir) in enumerate(methods):
        if method == "learned" and fn_dir is None:
            continue  # skip learned if no FusionNet provided
        port = udp_port(m_ix)  # 5001, 5002, 5003...
        udp_addr = f"239.0.0.1:{port}"
        out_dir = out_root / f"{method}"
        out_dir.mkdir(parents=True, exist_ok=False)

        # 1) Start logger
        logger_cmd = [
            sys.executable, "logger.py",
            "--udp", udp_addr,
            "--exp", exp,
            "--target", target,
            "--trackers", ",".join(trackers),
            "--method", ("learned" if method == "learned" else "gossip" if method == "gossip" else "grid"),
            "--ci_objective", "trace",
            "--ci_grid", "0.1",
            "--out", str(out_dir),
            "--timeout_ms", "300",
            "--fanout",
            "--planner", "eig"
        ]
        if method == "learned":
            logger_cmd += ["--fusionnet_dir", str(fn_dir)]
        logger_proc = spawn(logger_cmd, cwd=root)
        time.sleep(1.0)  # let logger join multicast before nodes start

        # 2) Start robot nodes for each tracker
        node_procs = []
        for rid in trackers:
            cmd = [
                sys.executable, "robot_node.py",
                "--id", rid,
                "--target", target,
                "--exp", exp,
                "--udp", udp_addr,
                *node_flags
            ]
            node_procs.append(spawn(cmd, cwd=root))

        # 3) Wait for nodes to finish; then wait for logger
        try:
            for p in node_procs:
                p.wait()
        finally:
            try:
                logger_proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                logger_proc.terminate()
                try:
                    logger_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger_proc.kill()

# ---------- summary aggregation ----------

def harvest_summaries(search_roots: List[Path], out_csv: Path) -> None:
    """
    Walk all provided roots, pick every 'summary.csv' and combine into one CSV with columns:
      exp_dir, rmse_x, rmse_y, rmse_z, rmse_3d, nees
    """
    import csv
    rows = []
    for root in search_roots:
        for p in root.rglob("summary.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(p)
                if df.empty: continue
                rec = df.iloc[0].to_dict()
                rec["exp_dir"] = str(p.parent.relative_to(root.parent))
                rows.append(rec)
            except Exception:
                # lightweight fallback
                with open(p, "r") as f:
                    lines = f.read().strip().splitlines()
                if len(lines) >= 2:
                    header = [h.strip() for h in lines[0].split(",")]
                    vals = [v.strip() for v in lines[1].split(",")]
                    d = dict(zip(header, vals))
                    d["exp_dir"] = str(p.parent.relative_to(root.parent))
                    rows.append(d)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["exp_dir","rmse_x","rmse_y","rmse_z","rmse_3d","nees"])
        w.writeheader()
        for r in rows:
            w.writerow({
                "exp_dir": r.get("exp_dir",""),
                "rmse_x": r.get("rmse_x",""),
                "rmse_y": r.get("rmse_y",""),
                "rmse_z": r.get("rmse_z",""),
                "rmse_3d": r.get("rmse_3d",""),
                "nees": r.get("nees",""),
            })
    print(f"[TABLE] Combined summaries -> {out_csv}  (n={len(rows)})")

# ---------- main orchestration ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.getcwd(), help="Repo root (where scripts live)")
    ap.add_argument("--expdir", default="data/three_robots", help="Experiments directory")
    ap.add_argument("--target", default="ifo003", help="Robot id to use as the TARGET in all runs")
    ap.add_argument("--trackers", nargs="+", default=["ifo001","ifo002"], help="Tracker ids to use (UDP + eval budgeting)")
    ap.add_argument("--tag", default="noLOS_cv_udp", help="A short tag to include in run folder name")
    ap.add_argument("--budget_k", type=int, default=2)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    os.chdir(root)
    expdir = (root / args.expdir).resolve()

    # Per-run unique root; nothing overwrites previous runs
    run_root = Path("runs") / f"{stamp()}_{args.tag}"
    run_root.mkdir(parents=True, exist_ok=False)
    cache_root = run_root / "collect_cache"
    cv_root    = run_root / "cv_nonzig"
    zig_root   = run_root / "zig_sections"
    udp_root   = run_root / "udp"
    models_root= run_root / "models"
    ds_root    = run_root / "datasets"

    print(f"[ENV] run_root={run_root}")
    print(f"[ENV] expdir={expdir}")
    print(f"[ENV] target={args.target} trackers={args.trackers}")

    all_exps, nonzig_exps, zig_exps = discover_experiments(expdir)
    if not nonzig_exps:
        raise RuntimeError("No non-zigzag experiments found after filtering.")

    # ---------- STEP 0: per-(exp,target) collect-only (never overwrites) ----------
    print("\n[STEP 0] Collect BiasNet+FusionNet datasets (collect-only, cached per (exp,target))")
    for exp in all_exps:
        outdir = cache_root / args.target / f"{exp}_{args.target}"
        if outdir.exists():
            # Respect "no overwrite": skip if something already there for this run,
            # otherwise collect to a new unique subdir under cache/target/<exp_*>
            print(f"  [SKIP collect] found {outdir}")
            continue
        ensure_collect(exp=exp, target=args.target, outdir=outdir, root=root)

    # ---------- STEP 1: Non-zigzag CV (leave-one-experiment-out) ----------
    print("\n[STEP 1] Non-zigzag LOEO CV")
    fold_idx = 0
    cv_model_dirs_per_fold: List[Dict[str, Path]] = []
    for test_exp in nonzig_exps:
        fold_idx += 1
        print(f"\n  [FOLD {fold_idx}/{len(nonzig_exps)}] TEST={test_exp}")
        train_exps = [e for e in nonzig_exps if e != test_exp]
        fold_dir   = cv_root / f"fold_{fold_idx}"
        ds_dir     = ds_root / f"cv_fold_{fold_idx}"
        # dataset
        build_dataset(cache_root, ds_dir, [args.target], train_exps)
        # train BN/FN
        bn_time_dir = train_biasnet(ds_dir, models_root / f"bn_fold{fold_idx}_time", root)
        bn_exp_dir  = bn_time_dir.parent / (bn_time_dir.name + "_byexp")  # sibling created in train_biasnet()
        fn_root_dir = train_fusionnet(ds_dir, models_root / f"fn_fold{fold_idx}", root)
        cv_model_dirs_per_fold.append({"bn_time": bn_time_dir, "bn_exp": bn_exp_dir, "fn_root": fn_root_dir})
        # eval scenarios on the held-out test_exp
        eval_scenarios(test_exp, args.target, cv_model_dirs_per_fold[-1], fold_dir, root,
                       budget_k=args.budget_k, run_planners=True, decentralized=True)

    # ---------- STEP 2: Zigzag sections ----------
    if zig_exps:
        print("\n[STEP 2] Zigzag sections")
        # B1) train on NONZIG, test on each zig
        ds_zgen = ds_root / "zig_generalization"
        build_dataset(cache_root, ds_zgen, [args.target], nonzig_exps)
        bn_time_dir = train_biasnet(ds_zgen, models_root / "bn_zgen_time", root)
        bn_exp_dir  = bn_time_dir.parent / (bn_time_dir.name + "_byexp")
        fn_root_dir = train_fusionnet(ds_zgen, models_root / "fn_zgen", root)
        for zt in zig_exps:
            eval_scenarios(zt, args.target,
                           {"bn_time": bn_time_dir, "bn_exp": bn_exp_dir, "fn_root": fn_root_dir},
                           zig_root / "generalization" / zt, root,
                           budget_k=args.budget_k, run_planners=True, decentralized=True)

        # B2) zigzag-only LOEO (optional but thorough)
        for i, ztest in enumerate(zig_exps, start=1):
            ztrain = [e for e in zig_exps if e != ztest]
            zfold_ds = ds_root / f"zig_fold_{i}"
            build_dataset(cache_root, zfold_ds, [args.target], ztrain)
            bn_time_dir = train_biasnet(zfold_ds, models_root / f"bn_zigfold{i}_time", root)
            bn_exp_dir  = bn_time_dir.parent / (bn_time_dir.name + "_byexp")
            fn_root_dir = train_fusionnet(zfold_ds, models_root / f"fn_zigfold{i}", root)
            eval_scenarios(ztest, args.target,
                           {"bn_time": bn_time_dir, "bn_exp": bn_exp_dir, "fn_root": fn_root_dir},
                           zig_root / f"zig_fold_{i}", root,
                           budget_k=args.budget_k, run_planners=True, decentralized=True)

    # ---------- STEP 3: UDP decentralized demos ----------
    print("\n[STEP 3] UDP decentralized demos (grid, gossip, learned)")
    # Use a solid mixed dataset (all nonzig) to train models once for UDP learned run
    ds_udp = ds_root / "udp_all_nonzig"
    build_dataset(cache_root, ds_udp, [args.target], nonzig_exps)
    bn_udp_dir = train_biasnet(ds_udp, models_root / "bn_udp_time", root)
    fn_udp_dir = train_fusionnet(ds_udp, models_root / "fn_udp", root) / "fusionnet_by_time"

    # Run UDP on one representative nonzig experiment (the last fold’s test works, else pick the first)
    test_for_udp = nonzig_exps[-1] if nonzig_exps else all_exps[0]
    udp_run(exp=test_for_udp,
            target=args.target,
            trackers=list(args.trackers),
            bn_dir=bn_udp_dir,
            fn_dir_time=fn_udp_dir,
            out_root=udp_root / f"{test_for_udp}_{args.target}",
            root=root)

    # ---------- STEP 4: Master summary table ----------
    print("\n[STEP 4] Aggregating summaries")
    harvest_summaries(
        [cv_root, zig_root, udp_root],
        out_csv=run_root / "all_summaries.csv"
    )

    print("\n===== DONE =====")
    print(f"Run root: {run_root}")
    print("Artifacts:")
    print("  - Collected data:     ", ds_root)
    print("  - Trained models:     ", models_root)
    print("  - CV evals (nonzig):  ", cv_root)
    print("  - Zigzag sections:    ", zig_root if zig_exps else "(no zig exps)")
    print("  - UDP demos:          ", udp_root)
    print("  - Master table:       ", run_root / 'all_summaries.csv')


if __name__ == "__main__":
    main()
