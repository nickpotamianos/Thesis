# -*- coding: utf-8 -*-
"""
Main entry‑point to run an experiment (online ISAM2 demo).
Now supports:
  •  --save-dir  : dump each time‑window graph / values for batch replay
  •  --csv-out   : export final ISAM2 poses to CSV
"""

from __future__ import annotations
import argparse, time, pathlib, gtsam

from sota.factor_graph.coordinator import FGCoordinator
from sota.factor_graph.utils_io  import ensure_dir, save_estimates   ### NEW
# --------------------------------------------------------------------------- #
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog = "fg_runner.py",
        description = "Realtime factor‑graph demo based on ISAM2"
    )
    p.add_argument("exp", help="experiment name / directory")
    p.add_argument("--max-sec", type=float, default=None,
                   help="wall‑clock limit (omit for full run)")
    p.add_argument("--save-dir", type=str, default=None,          ### NEW
                   help="dump every incremental graph/values here")
    p.add_argument("--csv-out", type=str, default=None,           ### NEW
                   help="CSV file to store final ISAM2 poses")
    return p

# --------------------------------------------------------------------------- #
def main() -> None:
    args = build_arg_parser().parse_args()

    if args.save_dir:
        args.save_dir = ensure_dir(args.save_dir)

    coord = FGCoordinator(args.exp, save_dir=args.save_dir)       ### EDIT
    t0 = time.time()
    coord.run(max_sec=args.max_sec)          # None → full timeline
    dt = time.time() - t0
    print(f"Finished in {dt:.2f}s")

    if args.csv_out:                         ### NEW
        save_estimates(coord.isam, args.csv_out)
        print(f"ISAM2 poses saved to {args.csv_out}")

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
