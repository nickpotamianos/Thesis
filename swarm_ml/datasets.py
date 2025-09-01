from __future__ import annotations
from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .features import (
    build_measurement_features,
    se_translation_from_matrix,
)
from .tagmap import (
    select_pair_rows,
    robust_range_aggregate,
    robust_tracker_sensor_position,
    robust_target_offset,
)
from .los_adapter import LOSAdapter, LOSConfig


class BiasNetDataset(Dataset):
    """
    Supervision: bias = measured_range - true_geometric_range.
    Each item: (features, bias) where features are built by build_measurement_features(...)
    """
    def __init__(self, samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = torch.tensor(s["features"], dtype=torch.float32)
        y = torch.tensor(s["bias"], dtype=torch.float32)
        # Optional: importance weight from pair variance (prefer lower R)
        w_val = 1.0
        try:
            R_pair = float(s.get("meta", {}).get("R_pair", 0.0))
            if np.isfinite(R_pair) and R_pair > 0:
                w_val = float(1.0 / np.sqrt(R_pair))
        except Exception:
            w_val = 1.0
        w = torch.tensor(w_val, dtype=torch.float32)
        return x, y, w


def _index_for_time(ts: np.ndarray, t: float) -> int:
    """
    Return index of t in ts. Assumes ts are exactly the query_timestamps used to build inputs.
    Falls back to nearest if exact match is missing.
    """
    idx = np.where(ts == t)[0]
    if idx.size > 0:
        return int(idx[0])
    # Safe nearest-neighbor fallback
    return int(np.argmin(np.abs(ts - t)))


def _gt_positions_from_T(gt_T_by_robot: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    """Extract 3D position sequences for every robot from SE(3)/SE_2(3) transforms."""
    out = {}
    for r, mats in gt_T_by_robot.items():
        out[r] = np.asarray([se_translation_from_matrix(M) for M in mats], dtype=float)
    return out


def build_biasnet_samples(
    exp_name: str,
    roles,  # swarm_ml.roles.Roles
    query_timestamps: np.ndarray,
    uwb_range_df: pd.DataFrame,
    gt_T_by_robot: Dict[str, List[np.ndarray]],
    tag_moment_arms,
    tag_map: Dict[str, List[int]],
    height_series: Optional[Dict[str, np.ndarray]] = None,
    los_adapter: Optional[LOSAdapter] = None,
    base_var: float = 0.35**2,
    pair_corr: float = 0.7,
    huber_delta: float = 0.8,
) -> List[Dict]:
    """
    Build per-link supervised samples:
      bias = z_agg - true_geometric_range

    Geometry for "true_geometric_range" is computed from **ground truth**:
      - tracker sensor world position using the tags that actually fired at time t
      - target tag offset in world using the target tags that fired at t
      - distance to the target's GT body center (consistent with runtime parameterization)

    Args:
        roles: Roles(target, trackers, tag_ids_by_robot=tag_map)
        query_timestamps: 1D array of times used throughout the run
        uwb_range_df: concatenated inter-robot UWB ranges with columns:
            ['timestamp','from_id','to_id','range','robot', ...]
        gt_T_by_robot: {robot -> list of SE(3) or SE_2(3) matrices} at query_timestamps
        tag_moment_arms: mapping from tag_id to arm (or nested by robot) from MILUV
        tag_map: {robot -> [tag ids]} inferred earlier
        height_series: optional per-robot height aligned to query_timestamps
        los_adapter: optional LOSAdapter to produce a LOS score per pair_df
    """
    if los_adapter is None:
        los_adapter = LOSAdapter(LOSConfig(verbose=False))

    gt_pos = _gt_positions_from_T(gt_T_by_robot)

    samples: List[Dict] = []
    tgt = roles.target
    trks = roles.trackers

    # Ensure fast lookup by timestamp
    # If your df is large, pre-grouping by timestamp is faster than boolean indexing
    uwb_by_t = dict(tuple(uwb_range_df.groupby("timestamp"))) if "timestamp" in uwb_range_df.columns else {}

    for t in query_timestamps:
        df_t = uwb_by_t.get(t, None)
        if df_t is None or df_t.empty:
            continue

        k = _index_for_time(query_timestamps, float(t))
        T_tgt = gt_T_by_robot[tgt][k]
        p_tgt = gt_pos[tgt][k]

        for trk in trks:
            # Keep rows that truly connect tracker tags to target tags
            df_pair = select_pair_rows(
                df_t,
                trk_tags=tag_map.get(trk, []),
                tgt_tags=tag_map.get(tgt, []),
            )
            if df_pair.empty:
                continue

            # Robustly aggregate multiple tag pairs at this t
            z_agg, R_pair, _ = robust_range_aggregate(
                df_pair, base_var=base_var, rho=pair_corr, huber_delta=huber_delta
            )

            # Tracker sensor world position using **GT** pose and the tags that actually fired
            T_trk = gt_T_by_robot[trk][k]
            p_trk_sens = robust_tracker_sensor_position(
                pair_df=df_pair,
                trk=trk,
                trk_tags=tag_map.get(trk, []),
                T_trk=T_trk,
                tag_moment_arms=tag_moment_arms,
                huber_delta=huber_delta
            )

            # Target tag offset in world using **GT** pose and the target tags that fired
            tgt_off_w = robust_target_offset(
                pair_df=df_pair,
                tgt_tags=tag_map.get(tgt, []),
                T_tgt=T_tgt,
                tag_moment_arms=tag_moment_arms,
                huber_delta=huber_delta
            )

            # Effective sensor position consistent with runtime parametrization
            eff_sensor_pos_gt = p_trk_sens - tgt_off_w

            # True geometric range (GT)
            true_range = float(np.linalg.norm(p_tgt - eff_sensor_pos_gt))

            # Optional LOS score (consistent with runtime)
            los_score = los_adapter.score(df_pair)  # may return None

            # Optional height difference (z_tgt - z_trk); if not provided, 0.0
            if height_series is not None and trk in height_series and tgt in height_series:
                dz = float(height_series[tgt][k] - height_series[trk][k])
            else:
                dz = 0.0

            # Build measurement features; no target_pred_pos for offline labels
            feat = build_measurement_features(
                tracker_pos=eff_sensor_pos_gt,
                target_pred_pos=None,
                uwb_range=float(z_agg),
                los_score=los_score,
                residual_hist=None,
                height_tracker=None,  # we directly pass height diff below
                height_target=None
            )
            # Append height difference and simple residual stats to the tail to
            # keep feature dimensionality aligned with your runtime builder
            feat = np.asarray(feat, dtype=float)
            # Final feature vector length remains consistent with runtime builder:
            # [uwb, bx, by, bz, dist, los, dz, mean_resid(0.0), var_resid(0.0)]
            if feat.shape[0] >= 6:
                feat[6] = dz  # overwrite height delta spot

            bias = float(z_agg - true_range)

            samples.append({
                "features": feat.tolist(),
                "bias": bias,
                "meta": {
                    "timestamp": float(t),
                    "tracker": trk,
                    "target": tgt,
                    "z_agg": float(z_agg),
                    "true_range": true_range,
                    "R_pair": float(R_pair),
                    "los": None if los_score is None else float(los_score),
                }
            })

    return samples


# ---------- simple utilities for JSONL IO (used by CLI trainers) ----------

def save_bias_samples_jsonl(samples: List[Dict], path: str) -> None:
    import json, os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


def load_bias_samples_jsonl(path: str) -> List[Dict]:
    import json, gzip
    out: List[Dict] = []
    if path.endswith('.gz'):
        fobj = gzip.open(path, 'rt')
    else:
        fobj = open(path, 'r')
    with fobj as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out
