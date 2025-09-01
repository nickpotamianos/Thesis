# tools/check_nis.py
import pandas as pd
import numpy as np
import sys

if len(sys.argv) != 2:
    print("Usage: python check_nis.py <nis_log.csv>")
    sys.exit(1)

df = pd.read_csv(sys.argv[1])
m = df['nis'].mean()
q = df['nis'].quantile([0.5, 0.9, 0.95, 0.99])
print(f"NIS mean: {m:.3f}")
print("NIS quantiles:")
print(q)
print(f"\nExpected for χ²₁: mean≈1.0, 50%≈0.45, 90%≈2.71, 95%≈3.84, 99%≈6.63")