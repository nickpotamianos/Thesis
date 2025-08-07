#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch re‑optimiser: merges all graph_*.g2o dumps produced by fg_runner.py
and runs a global Levenberg–Marquardt batch optimisation.
"""

from __future__ import annotations
import argparse, pathlib, tempfile, gtsam, re

VERTEX_RX = re.compile(r"^VERTEX.*\s+(\d+)\s")

# ------------------------------------------------------------------ #
def _dedup_g2o(src: pathlib.Path) -> pathlib.Path:
    """
    Return a temporary *.g2o* file identical to *src* but with duplicate
    VERTEX_* lines removed (first occurrence wins).
    """
    seen: set[str] = set()
    tmp = pathlib.Path(tempfile.mkstemp(suffix=".g2o")[1])

    with src.open("r") as fin, tmp.open("w") as fout:
        for line in fin:
            m = VERTEX_RX.match(line)
            if m:
                key = m.group(1)
                if key in seen:
                    continue          # skip duplicate vertex
                seen.add(key)
            fout.write(line)
    return tmp


def load_incremental(dir_: str):
    """
    Merge all *graph_*.g2o* files contained in *dir_* (each produced by a
    flush of fg_runner).  Duplicated vertex IDs inside ‑ or across ‑ files
    are tolerated; the newest Value wins.
    """
    dir_   = pathlib.Path(dir_)
    files  = sorted(dir_.glob("graph_*.g2o"))
    if not files:
        raise FileNotFoundError("no graph_*.g2o in", dir_)

    batch_graph  = gtsam.NonlinearFactorGraph()
    batch_values = gtsam.Values()

    for f in files:
        clean_f = _dedup_g2o(f)

        g_tmp, v_tmp = gtsam.readG2o(str(clean_f), is3D=True)
        batch_graph.push_back(g_tmp)

        for k in v_tmp.keys():
            # Try different value types that might be stored
            value = None
            try:
                value = v_tmp.atPose3(k)
            except RuntimeError:
                try:
                    value = v_tmp.atVector(k)
                except RuntimeError:
                    try:
                        value = v_tmp.atPoint3(k)
                    except RuntimeError:
                        continue  # Skip unknown value types
            
            if value is not None:
                if batch_values.exists(k):
                    batch_values.update(k, value)
                else:
                    batch_values.insert(k, value)

        clean_f.unlink()          # remove temporary file

    return batch_graph, batch_values
# ------------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dump_dir", help="directory produced by fg_runner.py --save-dir")
    ap.add_argument("--result-file", default="batch_result.values",
                    help="where to store the optimised Values()")
    args = ap.parse_args()

    graph, initial = load_incremental(args.dump_dir)
    print(f"Loaded {graph.size()} factors, {initial.size()} variables")

    prm = gtsam.LevenbergMarquardtParams()
    prm.setVerbosityLM("SUMMARY")
    opt = gtsam.LevenbergMarquardtOptimizer(graph, initial, prm)
    result = opt.optimizeSafely()

    out = pathlib.Path(args.result_file)
    try:                                    # if Values.save() is available
        result.save(str(out))
        print("Optimised Values written to", out.resolve())
    except AttributeError:                  # fallback – always works
        gtsam.writeG2o(gtsam.NonlinearFactorGraph(), result,
                       str(out.with_suffix(".g2o")))
        print("Optimised Values written to", out.with_suffix(".g2o").resolve())

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    main()
