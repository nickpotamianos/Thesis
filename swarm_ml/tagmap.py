# swarm_ml/tagmap.py
from typing import Dict, List
import pandas as pd
import numpy as np

def infer_tag_ids_by_robot(uwb_range_df: pd.DataFrame,
                           robots: List[str],
                           top_n: int = 2) -> Dict[str, List[int]]:
    """
    Infer which numeric tag IDs belong to which robot by frequency analysis.
    We look at all rows where a given robot produced the measurement and count
    how often each tag id appears in from_id/to_id. Return top_n ids per robot.
    """
    tag_map: Dict[str, List[int]] = {}
    if "robot" not in uwb_range_df.columns:
        # caller must annotate before calling
        for r in robots:
            tag_map[r] = []
        return tag_map

    for r in robots:
        df_r = uwb_range_df[uwb_range_df["robot"] == r]
        if df_r.empty:
            tag_map[r] = []
            continue
        counts = pd.concat([df_r["from_id"].value_counts(),
                            df_r["to_id"].value_counts()], axis=1).fillna(0).sum(axis=1)
        counts = counts.sort_values(ascending=False)
        tag_map[r] = [int(i) for i in list(counts.index[:top_n])]
    return tag_map

def select_pair_rows(df_t: pd.DataFrame,
                     trk_tags: List[int],
                     tgt_tags: List[int]) -> pd.DataFrame:
    """
    Keep only rows that connect any tracker tag to any target tag (either direction).
    """
    if df_t.empty or not trk_tags or not tgt_tags:
        return df_t.iloc[0:0]  # empty df
    m = df_t[
        ((df_t["from_id"].isin(trk_tags)) & (df_t["to_id"].isin(tgt_tags))) |
        ((df_t["from_id"].isin(tgt_tags)) & (df_t["to_id"].isin(trk_tags)))
    ]
    return m

def robust_range_aggregate(pair_df: pd.DataFrame,
                           base_var: float,
                           rho: float = 0.7,
                           huber_delta: float = 0.8,
                           use_dispersion: bool = True,
                           disp_tau: float = 0.20,
                           disp_gain: float = 0.6) -> tuple[float, float, dict]:
    """
    Robustly aggregate multiple tag-pair ranges for one tracker↔target at time t.
    - Location: Huber M-estimator (delta in meters).
    - Variance: base_var / m_eff, with m_eff = m / (1 + (m-1)*rho), rho in [0,1].
    Returns: (z_agg, R_eff, meta)
    """
    zs = pair_df["range"].to_numpy(dtype=float)
    m = zs.size
    med = float(np.median(zs))
    # Huber step
    r = zs - med
    w = np.ones_like(r)
    mask = np.abs(r) > huber_delta
    w[mask] = huber_delta / np.abs(r[mask])
    z_agg = float(np.sum(w * zs) / np.sum(w))

    # Effective sample size with correlation rho
    m_eff = m / (1.0 + (m - 1.0) * rho)
    m_eff = max(1.0, m_eff)  # never below 1

    R_eff = base_var / m_eff
    meta = {"m": int(m), "m_eff": float(m_eff)}
    # Optional dispersion-aware inflation using IQR
    if use_dispersion:
        if m >= 3:
            q25, q75 = np.percentile(zs, [25, 75])
            iqr = float(max(1e-9, q75 - q25))
        elif m == 2:
            # IQR-like proxy for two points
            iqr = float(abs(zs[1] - zs[0]))
        else:
            iqr = 0.0
        if iqr > 0.0:
            infl = 1.0 + float(disp_gain) * float(min(1.0, iqr / max(1e-9, disp_tau)))
            R_eff *= infl
            meta.update({"iqr": iqr, "R_infl": infl})
    return z_agg, float(R_eff), meta

def robust_tracker_sensor_position(pair_df: pd.DataFrame,
                                   trk: str,
                                   trk_tags: List[int],
                                   T_trk: np.ndarray,
                                   tag_moment_arms,
                                   huber_delta: float = 0.8) -> np.ndarray:
    """
    Compute a Huber-weighted average of the world positions of the tracker tags
    that actually fired at this timestamp. If anything fails, return the robot center.
    """
    from swarm_ml.features import se_rotation_from_matrix, se_translation_from_matrix

    if pair_df is None or pair_df.empty:
        return se_translation_from_matrix(T_trk)

    zs = pair_df["range"].to_numpy(dtype=float)
    med = float(np.median(zs))
    r = zs - med
    w = np.ones_like(r)
    mask = np.abs(r) > huber_delta
    w[mask] = huber_delta / np.abs(r[mask])

    R = se_rotation_from_matrix(T_trk)
    p = se_translation_from_matrix(T_trk)

    positions = []
    w_list = []
    for (_, row), wi in zip(pair_df.iterrows(), w):
        # Determine which tracker tag contributed in this row
        tid = None
        f = int(row["from_id"]); t = int(row["to_id"])
        if f in trk_tags: tid = f
        elif t in trk_tags: tid = t

        if tid is None:
            continue
        # Lookup moment arm for this tag; fall back to robot center
        arm = None
        if isinstance(tag_moment_arms, dict):
            if tid in tag_moment_arms:
                arm = np.asarray(tag_moment_arms[tid], dtype=float).reshape(3)
            else:
                for v in tag_moment_arms.values():
                    if isinstance(v, dict) and tid in v:
                        arm = np.asarray(v[tid], dtype=float).reshape(3)
                        break
        if arm is None:
            pos = p
        else:
            pos = p + R @ arm
        positions.append(pos)
        w_list.append(wi)

    if not positions:
        return p
    P = np.vstack(positions)
    wv = np.asarray(w_list, dtype=float).reshape(-1, 1)
    wv /= (wv.sum() + 1e-12)
    return (wv * P).sum(axis=0)

def robust_target_offset(pair_df: pd.DataFrame,
                         tgt_tags: List[int],
                         T_tgt: np.ndarray,
                         tag_moment_arms,
                         huber_delta: float = 0.8) -> np.ndarray:
    """
    Return world-frame offset vector R_tgt @ arm(tag) for the TARGET tag(s) that fired
    at this timestamp. Uses Huber weights over all tag-pair rows at t.
    If unknown, returns zeros(3).
    """
    from swarm_ml.features import se_rotation_from_matrix

    if pair_df is None or pair_df.empty or not tgt_tags:
        return np.zeros(3)

    zs = pair_df["range"].to_numpy(dtype=float)
    med = float(np.median(zs))
    r = zs - med
    w = np.ones_like(r)
    mask = np.abs(r) > huber_delta
    w[mask] = huber_delta / np.abs(r[mask])

    R_t = se_rotation_from_matrix(T_tgt)

    # Collect offset candidates (R@arm) for any target tag that fired
    offsets, w_list = [], []
    for (_, row), wi in zip(pair_df.iterrows(), w):
        tid = None
        f = int(row["from_id"]); t = int(row["to_id"])
        if f in tgt_tags: tid = f
        elif t in tgt_tags: tid = t
        if tid is None:
            continue

        # Lookup arm for target tag id (allow nested dicts)
        arm = None
        if isinstance(tag_moment_arms, dict):
            if tid in tag_moment_arms:
                arm = np.asarray(tag_moment_arms[tid], dtype=float).reshape(3)
            else:
                for v in tag_moment_arms.values():
                    if isinstance(v, dict) and tid in v:
                        arm = np.asarray(v[tid], dtype=float).reshape(3)
                        break
        if arm is None:
            continue

        offsets.append(R_t @ arm)
        w_list.append(wi)

    if not offsets:
        return np.zeros(3)

    O = np.vstack(offsets)
    wv = np.asarray(w_list, dtype=float).reshape(-1, 1)
    wv /= (wv.sum() + 1e-12)
    return (wv * O).sum(axis=0)