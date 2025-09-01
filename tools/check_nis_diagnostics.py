# tools/check_nis_diagnostics.py
import pandas as pd, numpy as np, sys

if len(sys.argv) != 2:
    print("Usage: python tools/check_nis_diagnostics.py <nis_log.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])  # expects columns: timestamp,tracker,nis,S,R_eff
if not set(["timestamp","tracker","nis","S","R_eff"]).issubset(df.columns):
    raise SystemExit("nis_log.csv must have columns: timestamp, tracker, nis, S, R_eff")

df["HPH"] = df["S"] - df["R_eff"]
df = df.replace([np.inf, -np.inf], np.nan).dropna()

print(f"N = {len(df)} updates")
print(f"mean NIS = {df.nis.mean():.3f}")

print("\nFractional contributions to S:")
print(f"  mean R_eff/S   = {(df.R_eff/df.S).mean():.3f}")
print(f"  mean HPH/S     = {(df.HPH/df.S).mean():.3f}")

print("\nTracker-wise mean NIS and R_eff/S:")
print(df.groupby('tracker').agg(
    mean_NIS=('nis','mean'),
    mean_Rfrac=('R_eff', lambda s: (s/df.loc[s.index,'S']).mean())
))

# Heuristic suggestion for R scaling toward mean NIS ~ 1 when R dominates
rfrac = (df.R_eff/df.S).mean()
mnis = df.nis.mean()
if rfrac > 0.5:
    H = df.HPH.mean(); R = df.R_eff.mean()
    alpha = (mnis*(H+R) - H)/R
    print(f"\nHeuristic R scale to hit mean NIS≈1 (R-dominated): alpha ≈ {alpha:.3f}")
else:
    print("\nS is mostly HPH (state covariance) -> reduce Q (sigma_a) rather than R.")
