#swarm-ml/los_classification.py
import numpy as np
import pandas as pd


def _score_from_iqr(zs: np.ndarray) -> float:
    """
    Map intra‑timestep dispersion to a LOS probability in [0.05, 0.95].
    Works with small m by shrinking toward 0.5 (including m=2).
    """
    zs = np.asarray(zs, dtype=float).reshape(-1)
    m = zs.size
    if m < 2:
        return 0.5  # truly too little data

    # Use an "IQR-like" spread for small m as well
    if m == 2:
        iqr = abs(float(zs[1] - zs[0]))
        # gentle logistic and heavy shrink so we don't overreact with 2 pts
        k, x0 = 6.0, 0.12
        p_raw = 1.0 / (1.0 + np.exp(k * (iqr - x0)))
        p_raw = float(np.clip(p_raw, 0.05, 0.95))
        shrink = 0.25  # 25% of the deviation from 0.5 when m=2
        return 0.5 + shrink * (p_raw - 0.5)

    # Use an "IQR-like" spread for small m as well
    if m == 2:
        iqr = abs(float(zs[1] - zs[0]))
        # gentle logistic and heavy shrink so we don't overreact with 2 pts
        k, x0 = 6.0, 0.12
        p_raw = 1.0 / (1.0 + np.exp(k * (iqr - x0)))
        p_raw = float(np.clip(p_raw, 0.05, 0.95))
        shrink = 0.25  # 25% of the deviation from 0.5 when m=2
        return 0.5 + shrink * (p_raw - 0.5)

    q25, q75 = np.percentile(zs, [25, 75])
    iqr = max(1e-6, float(q75 - q25))

    # Logistic: small IQR → high LOS; parameters slightly relaxed for small m
    # (wider "good" region and gentler slope)
    if m < 5:
        k, x0 = 6.0, 0.12
    else:
        k, x0 = 8.0, 0.15

    p_raw = 1.0 / (1.0 + np.exp(k * (iqr - x0)))
    p_raw = float(np.clip(p_raw, 0.05, 0.95))

    # Small‑sample shrinkage toward neutral so 3–4 samples don't over‑swing.
    # m=3→33% of the deviation from 0.5, m=4→66%, m≥5→100%.
    shrink = min(1.0, (m - 2) / 3.0)
    return 0.5 + shrink * (p_raw - 0.5)


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

