#tools/autotune_r.py
import argparse, pandas as pd, numpy as np

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("nis_log", help="outputs/.../nis_log.csv")
    args = ap.parse_args()
    df = pd.read_csv(args.nis_log)
    mu = df["nis"].mean()
    # If R dominates S: shrink R by factor mu (clip to 0.25..1.5)
    r_alpha = float(np.clip(mu, 0.25, 1.5))
    print(f"mean NIS={mu:.3f} → recommended R scale ≈ {r_alpha:.3f}")
    print("Try: --uwb_std NEW_STD where NEW_STD = OLD_STD * sqrt(r_alpha)")

