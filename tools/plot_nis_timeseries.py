#tools/plot_nis_timeseries.py
import argparse, pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nis_csv", help="outputs/.../nis_timeseries.csv or nis_log.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.nis_csv)
    if "roll_mean_global" not in df.columns:
        df = df.sort_values("timestamp")
        df["roll_mean_global"] = df["nis"].rolling(200, min_periods=1).mean()
    for k, g in df.groupby("tracker"):
        plt.plot(g["timestamp"], g["roll_mean_global"], label=k)
    for q, y in [("q50",0.455),("q90",2.706),("q95",3.841),("q99",6.635)]:
        plt.axhline(y, linestyle="--", linewidth=0.8)
    plt.legend(); plt.xlabel("t"); plt.ylabel("rolling mean NIS")
    plt.title("Per-tracker rolling NIS (200-win)")
    plt.show()

