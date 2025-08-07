#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert a gtsam.Values file (from batch_optimize.py) into one CSV per robot,
plus a single combined CSV identical in format to utils_io.save_estimates().
"""

from __future__ import annotations
import argparse, csv, pathlib, re, gtsam
import numpy as np

# ---------- helpers --------------------------------------------------------- #
def vec3(v):  # Point3 or ndarray → ndarray(3,)
    return np.asarray([v.x(), v.y(), v.z()]) if hasattr(v, "x") else np.asarray(v)

def quat_xyzw(q):
    # ndarray may be [w,x,y,z]; rearrange to x,y,z,w
    if hasattr(q, "x"):
        return np.array([q.x(), q.y(), q.z(), q.w()])
    q = np.asarray(q)
    return q[1:], q[0] if q.shape == (4,) else q

def get_symbol(k: int) -> tuple[str,int]:
    s = gtsam.Symbol(k)
    try:
        _c = s.chr()
        c = chr(_c) if isinstance(_c, int) else _c
        # If we get a null character, it means this is from G2O format
        # which doesn't preserve symbol chars. Try to infer from key range.
        if c == '\x00' or ord(c) == 0:
            # Simple heuristic: assume robots P, Q, R based on ranges
            if k < 25:
                c = 'P'  # Robot 0
            elif k < 50:
                c = 'Q'  # Robot 1  
            else:
                c = 'R'  # Robot 2
        return c, s.index()
    except AttributeError:
        # Fallback for older GTSAM versions
        if k < 25:
            return 'P', k
        elif k < 50:
            return 'Q', k - 25
        else:
            return 'R', k - 50

# ---------- main ------------------------------------------------------------ #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("values_file", help="*.values produced by batch_optimize.py")
    ap.add_argument("--out-dir", default="out/batch_csv", help="directory to store CSVs")
    ap.add_argument("--plot", action="store_true", help="quick matplotlib XY plot")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    values = gtsam.Values()
    
    # Handle both .values and .g2o files
    if args.values_file.endswith('.g2o'):
        # Load from G2O file
        _, values = gtsam.readG2o(args.values_file, is3D=True)
    else:
        # Try to load from Values file
        try:
            values.load(args.values_file)
        except AttributeError:
            print(f"Error: Cannot load {args.values_file}. Values.load() not available.")
            return
    combined_path = out_dir / "all_robots.csv"

    writers = {}        # robot‑letter → csv.writer
    csv_files = {}      # to close at the end

    def get_writer(letter: str):
        if letter not in writers:
            f = (out_dir / f"robot_{letter}.csv").open("w", newline="")
            csv_files[letter] = f
            w = csv.writer(f); writers[letter] = w
            w.writerow(["idx","x","y","z","qx","qy","qz","qw"])
        return writers[letter]

    with combined_path.open("w", newline="") as f_all:
        w_all = csv.writer(f_all)
        w_all.writerow(["key","sym","idx","x","y","z","qx","qy","qz","qw"])

        for k in values.keys():
            try:
                pose = values.atPose3(k)
            except RuntimeError:
                continue

            sym, idx = get_symbol(k)
            t = vec3(pose.translation())
            qx,qy,qz,qw = quat_xyzw(pose.rotation().toQuaternion())

            row_all = [k, sym, idx, float(t[0]), float(t[1]), float(t[2]), 
                      float(qx), float(qy), float(qz), float(qw)]
            try:
                w_all.writerow(row_all)
            except Exception as e:
                print(f"Error writing row: {row_all}")
                print(f"Error: {e}")
                continue

            # per‑robot file (store idx + pose only)
            try:
                get_writer(sym).writerow([idx, float(t[0]), float(t[1]), float(t[2]), 
                                        float(qx), float(qy), float(qz), float(qw)])
            except Exception as e:
                print(f"Error writing robot row: {[idx, float(t[0]), float(t[1]), float(t[2]), float(qx), float(qy), float(qz), float(qw)]}")
                print(f"Error: {e}")
                continue

    for f in csv_files.values():
        f.close()
    print(f"[export_values] wrote {combined_path} and {len(csv_files)} per‑robot CSVs")

    # optional quick‑look plot ------------------------------------------------
    if args.plot:
        import matplotlib.pyplot as plt
        for letter, file in csv_files.items():
            data = np.loadtxt(file.name, delimiter=",", skiprows=1)
            plt.plot(data[:,1], data[:,2], label=f"robot {letter}")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(); plt.xlabel("x [m]"); plt.ylabel("y [m]")
        plt.title("Batch‑optimised trajectories")
        plt.show()

if __name__ == "__main__":
    main()
