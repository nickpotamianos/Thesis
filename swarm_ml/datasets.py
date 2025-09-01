# swarm_ml/datasets.py
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
    Each item: (features, bias, weight)
    Weight defaults to 1.0, optionally derived from pair variance to up‑weight reliable pairs.
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

    True geometric range uses GT:
      - robust tracker sensor world position (tags that actually fired at t)
      - robust target tag offset (tags that fired at t)
      - distance to target GT body center (consistent with runtime parametrization)
    """
    if los_adapter is None:
        los_adapter = LOSAdapter(LOSConfig(verbose=False))

    gt_pos = _gt_positions_from_T(gt_T_by_robot)

    samples: List[Dict] = []
    tgt = roles.target
    trks = roles.trackers

    # Pre-group for faster per‑t timestamp access
    uwb_by_t = dict(tuple(uwb_range_df.groupby("timestamp"))) if "timestamp" in uwb_range_df.columns else {}

    for t in query_timestamps:
        df_t = uwb_by_t.get(t, None)
        if df_t is None or df_t.empty:
            continue

        k = _index_for_time(query_timestamps, float(t))
        T_tgt = gt_T_by_robot[tgt][k]
        p_tgt = gt_pos[tgt][k]

        for trk in trks:
            # rows that truly connect tracker tags to target tags
            df_pair = select_pair_rows(
                df_t,
                trk_tags=tag_map.get(trk, []),
                tgt_tags=tag_map.get(tgt, []),
            )
            if df_pair.empty:
                continue

            # robust aggregation across tag-pairs at t
            z_agg, R_pair, _ = robust_range_aggregate(
                df_pair, base_var=base_var, rho=pair_corr, huber_delta=huber_delta
            )

            # tracker sensor world position (GT) and target offset (GT)
            T_trk = gt_T_by_robot[trk][k]
            p_trk_sens = robust_tracker_sensor_position(
                pair_df=df_pair,
                trk=trk,
                trk_tags=tag_map.get(trk, []),
                T_trk=T_trk,
                tag_moment_arms=tag_moment_arms,
                huber_delta=huber_delta
            )
            tgt_off_w = robust_target_offset(
                pair_df=df_pair,
                tgt_tags=tag_map.get(tgt, []),
                T_tgt=T_tgt,
                tag_moment_arms=tag_moment_arms,
                huber_delta=huber_delta
            )
            eff_sensor_pos_gt = p_trk_sens - tgt_off_w

            # True geometric range
            true_range = float(np.linalg.norm(p_tgt - eff_sensor_pos_gt))

            # Optional LOS score
            los_score = los_adapter.score(df_pair)  # may be None

            # Optional height difference (z_tgt - z_trk)
            if height_series is not None and trk in height_series and tgt in height_series:
                dz = float(height_series[tgt][k] - height_series[trk][k])
            else:
                dz = 0.0

            # Build feature vector (keep layout consistent with runtime builder)
            feat = build_measurement_features(
                tracker_pos=eff_sensor_pos_gt,
                target_pred_pos=None,
                uwb_range=float(z_agg),
                los_score=los_score,
                residual_hist=None,
                height_tracker=None,
                height_target=None
            )
            feat = np.asarray(feat, dtype=float)
            if feat.shape[0] >= 7:
                feat[6] = dz  # overwrite 'dz' slot if present

            samples.append({
                "features": feat.tolist(),
                "bias": float(z_agg - true_range),
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
    """
    Load BiasNet samples from either .jsonl or .jsonl.gz.
    Also auto-fallback across extensions if the given file is missing.
    """
    import json, gzip, os

    def _resolve(p: str) -> str:
        if os.path.exists(p):
            return p
        if p.endswith(".jsonl") and os.path.exists(p + ".gz"):
            return p + ".gz"
        if p.endswith(".jsonl.gz") and os.path.exists(p[:-3]):
            return p[:-3]
        raise FileNotFoundError(
            f"Bias samples file not found. Tried: {p}, "
            f"{p + '.gz' if p.endswith('.jsonl') else ''}, "
            f"{p[:-3] if p.endswith('.gz') else ''}"
        )

    path = _resolve(path)
    out: List[Dict] = []
    open_fn = gzip.open if path.endswith(".gz") else open
    with open_fn(path, "rt") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
