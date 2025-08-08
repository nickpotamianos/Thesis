#!/usr/bin/env python3
import argparse, numpy as np, pandas as pd, matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="poses_fixed.csv from save_estimates()")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    plt.figure()
    for sym, g in df.groupby("sym"):
        g = g.sort_values("idx")
        plt.plot(g["x"], g["y"], marker=".", label=f"Robot {sym}")
    # anchors (from your dataset description)
    anchors = np.array([
        [ 3.27382739,  3.46404736, 1.80933093],
        [ 3.18638696,  0.27394485, 1.58848535],
        [ 2.85050024, -2.92305688, 1.89742041],
        [-2.49763452, -3.50182031, 1.77309119],
        [-2.95793311,  0.61284192, 1.65714209],
        [-2.73467651,  3.65854248, 1.89025464],
    ])
    plt.scatter(anchors[:,0], anchors[:,1], c="r", s=40, marker="s", label="Anchors")
    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlabel("x [m]"); plt.ylabel("y [m]")
    plt.title("Incremental ISAM2 Trajectories")
    plt.show()

if __name__ == "__main__":
    main()
