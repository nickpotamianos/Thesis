#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, csv, pathlib, gtsam, numpy as np

def vec3(v):  # Point3 or ndarray → ndarray(3,)
    return np.asarray([v.x(), v.y(), v.z()]) if hasattr(v, "x") else np.asarray(v, dtype=float)

def quat_xyzw(q):
    if hasattr(q, "x"):
        return (q.x(), q.y(), q.z(), q.w())
    q = np.asarray(q, dtype=float).reshape(4)
    # assume [w,x,y,z] if ndarray
    return (q[1], q[2], q[3], q[0])

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("values_file", help="batch_result.values or .txt fallback")
    ap.add_argument("--out-dir", default="out/batch_csv", help="directory to store CSVs")
    ap.add_argument("--plot", action="store_true", help="quick matplotlib XY plot")
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    values = gtsam.Values()
    # try .values first
    try:
        values.load(args.values_file)
    except Exception:
        # load our .txt fallback
        from sota.factor_graph.utils_io import _values_load_txt
        values = _values_load_txt(pathlib.Path(args.values_file))

    combined_path = out_dir / "all_robots.csv"
    writers, csv_files = {}, {}

    def writer_for(letter: str):
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

            sym = gtsam.Symbol(k)
            try:
                _c = sym.chr(); letter = chr(_c) if isinstance(_c, int) else _c
                idx = sym.index()
            except AttributeError:
                # very old builds
                s = str(sym); import re
                m = re.search(r"([A-Za-z])\s*,?\s*(\d+)", s)
                letter, idx = (m.group(1), int(m.group(2))) if m else ("?", 0)

            t = vec3(pose.translation())
            qx,qy,qz,qw = quat_xyzw(pose.rotation().toQuaternion())

            w_all.writerow([k, letter, idx, *map(float, (t[0],t[1],t[2],qx,qy,qz,qw))])
            writer_for(letter).writerow([idx, *map(float, (t[0],t[1],t[2],qx,qy,qz,qw))])

    for f in csv_files.values():
        f.close()
    print(f"[export_values] wrote {combined_path} and {len(csv_files)} per‑robot CSVs")

    if args.plot:
        import numpy as np, matplotlib.pyplot as plt
        for letter, fh in csv_files.items():
            data = np.loadtxt(fh.name, delimiter=",", skiprows=1)
            if data.ndim == 1 and data.size == 0:  # empty
                continue
            plt.plot(data[:,1], data[:,2], label=f"robot {letter}")
        plt.gca().set_aspect("equal", adjustable="box")
        plt.legend(); plt.xlabel("x [m]"); plt.ylabel("y [m]")
        plt.title("Batch‑optimised trajectories")
        plt.show()

if __name__ == "__main__":
    main()