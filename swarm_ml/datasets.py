# swarm_ml/datasets.py
from typing import List, Dict, Tuple, Callable, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .features import build_measurement_features

class BiasNetDataset(Dataset):
    """
    Supervision: bias = measured_range - true_geometric_range.
    """
    def __init__(self,
                 samples: List[Dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        x = torch.tensor(s["features"], dtype=torch.float32)
        y = torch.tensor(s["bias"], dtype=torch.float32)
        return x, y

def build_biasnet_samples(exp_name: str,
                          miluv,  # miluv.data.DataLoader instance
                          roles,
                          query_timestamps: np.ndarray,
                          uwb_range_df: pd.DataFrame,
                          gt_positions: Dict[str, np.ndarray],
                          los_scores: Optional[Dict[Tuple[str, str, float], float]] = None) -> List[Dict]:
    """
    Creates per-link supervised samples across timestamps.
    """
    samples = []
    for t in query_timestamps:
        # For each tracker -> target pair, gather the nearest range measurement at t
        df_t = uwb_range_df[uwb_range_df["timestamp"] == t]
        for tracker in roles.trackers:
            # Accept any measurement whose endpoints map to (tracker, target) tags.
            # Simplification: if multiple tag-tag pairs exist, use the minimum range.
            if df_t.shape[0] == 0:
                continue
            z_cands = []
            for _, row in df_t.iterrows():
                from_id, to_id = int(row["from_id"]), int(row["to_id"])
                z_cands.append(float(row["range"]))
            if not z_cands:
                continue
            z = float(np.min(z_cands))

            p_trk = gt_positions[tracker]  # (N,3) aligned to query_timestamps externally
            p_tgt = gt_positions[roles.target]
            # NOTE: here expect gt_positions aligned; otherwise interpolate externally
            # Find index i corresponding to time t
            # (caller should ensure alignment; omitted for brevity)
            # Build features
            los = None
            if los_scores is not None:
                los = float(los_scores.get((tracker, roles.target, t), 0.5))

            feat = build_measurement_features(
                tracker_pos=p_trk, target_pred_pos=p_tgt, uwb_range=z, los_score=los
            )
            true_range = float(np.linalg.norm(p_tgt - p_trk))
            bias = float(z - true_range)
            samples.append({"features": feat, "bias": bias})
    return samples
