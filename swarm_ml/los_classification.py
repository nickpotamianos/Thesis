import numpy as np
import pandas as pd


def _score_from_iqr(zs: np.ndarray) -> float:
    """Map intra-timestep dispersion to a LOS probability in [0.05, 0.95]."""
    zs = np.asarray(zs, dtype=float).reshape(-1)
    if zs.size < 5:
        return 0.5
    q25, q75 = np.percentile(zs, [25, 75])
    iqr = max(1e-6, float(q75 - q25))
    # Logistic: small IQR -> high LOS; midpoint around 0.15 m
    k, x0 = 8.0, 0.15
    p = 1.0 / (1.0 + np.exp(k * (iqr - x0)))
    return float(np.clip(p, 0.05, 0.95))


def predict_los_probability(obj) -> float:
    """
    Batch entrypoint for LOSAdapter. Accepts a DataFrame with a 'range' column
    or a 1D array of ranges.
    """
    if isinstance(obj, pd.DataFrame):
        zs = obj["range"].to_numpy(dtype=float)
    else:
        zs = np.asarray(obj, dtype=float).reshape(-1)
    return _score_from_iqr(zs)


def predict_row(row: dict) -> float:
    """Optional row-wise fallback (neutral-ish)."""
    try:
        z = float(row.get("range", np.nan))
        if not np.isfinite(z):
            return 0.5
    except Exception:
        return 0.5
    return 0.6

