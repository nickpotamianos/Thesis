from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import os, json
import numpy as np
import pandas as pd

from .features import se_translation_from_matrix, build_measurement_features
from .tagmap import robust_tracker_sensor_position, robust_target_offset
from .los_adapter import LOSAdapter, LOSConfig

@dataclass
class FusionSnap:
    timestamp: float
    order: List[str]                   # tracker ids order
    X: np.ndarray                      # (N_nodes, d) node features (var_pos, reliability, z_agg, R_eff)
    mus: np.ndarray                    # (N_nodes, 6) local posteriors mu_i
    Ps: np.ndarray                     # (N_nodes, 6, 6) local posteriors P_i
    gt_pos: np.ndarray                 # (3,)
    los: Optional[Dict[str, float]] = None

    def to_json(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "order": list(self.order),
            "X": self.X.tolist(),
            "mus": self.mus.tolist(),
            "Ps": self.Ps.tolist(),
            "gt_pos": self.gt_pos.tolist(),
            "los": None if self.los is None else {k: float(v) for k, v in self.los.items()}
        }


class SnapshotCollector:
    """
    Collects:
      - BiasNet samples (features + bias labels)
      - FusionNet snaps (per-timestep node features + local posteriors + GT target position)
    Saves JSONL files under the experiment out directory.
    """
    def __init__(self,
                 query_timestamps: np.ndarray,
                 roles,
                 tag_map: Dict[str, List[int]],
                 gt_T_by_robot: Dict[str, List[np.ndarray]],
                 tag_moment_arms,
                 base_var: float,
                 pair_corr: float,
                 huber_delta: float = 0.8,
                 los_verbose: bool = False):
        self.ts = np.asarray(query_timestamps, dtype=float)
        self.roles = roles
        self.tag_map = tag_map
        self.gt_T = gt_T_by_robot
        self.tag_moment_arms = tag_moment_arms
        self.base_var = float(base_var)
        self.pair_corr = float(pair_corr)
        self.huber_delta = float(huber_delta)
        self.los_adapter = LOSAdapter(LOSConfig(verbose=los_verbose))

        # precompute GT positions per robot
        self.gt_pos = {r: np.asarray([se_translation_from_matrix(T) for T in mats], dtype=float)
                       for r, mats in gt_T_by_robot.items()}

        self._bias_jsonl: List[Dict[str, Any]] = []
        self._fuse_jsonl: List[Dict[str, Any]] = []

    # ---- BiasNet collection (called from measurement path) ----
    def add_bias_sample(self,
                        i: int,
                        trk: str,
                        pair_df: pd.DataFrame,
                        z_agg: float,
                        eff_sensor_pos_used: np.ndarray,
                        los_score: Optional[float]) -> None:
        """
        Log a sample consistent with your runtime geometry (eff_sensor_pos used).
        Supervision uses GT target position at i.
        """
        t = float(self.ts[i])
        tgt = self.roles.target
        p_tgt = self.gt_pos[tgt][i]
        true_range = float(np.linalg.norm(p_tgt - eff_sensor_pos_used))

        feat = build_measurement_features(
            tracker_pos=np.asarray(eff_sensor_pos_used, dtype=float),
            target_pred_pos=None,
            uwb_range=float(z_agg),
            los_score=None if los_score is None else float(los_score),
        )
        # keep feature layout identical to runtime builder (fills optional slots with zeros)
        feat = np.asarray(feat, dtype=float)

        self._bias_jsonl.append({
            "features": feat.tolist(),
            "bias": float(z_agg - true_range),
            "meta": {
                "timestamp": t,
                "tracker": trk,
                "target": tgt,
                "z_agg": float(z_agg),
                "true_range": true_range,
                "los": None if los_score is None else float(los_score)
            }
        })

    # ---- FusionNet collection (called after building parts/node_feats, before fusion) ----
    def add_fusion_snap(self,
                        i: int,
                        parts: Dict[str, Tuple[np.ndarray, np.ndarray]],
                        node_feats: Dict[str, np.ndarray]) -> None:
        t = float(self.ts[i])
        order = list(parts.keys())  # keep the runtime order
        X = np.vstack([node_feats[k].reshape(-1) for k in order])
        mus = np.vstack([parts[k][0].reshape(1, -1) for k in order])
        Ps = np.stack([parts[k][1] for k in order], axis=0)
        gt_pos = self.gt_pos[self.roles.target][i][:3]

        snap = FusionSnap(t, order, X, mus, Ps, gt_pos)
        self._fuse_jsonl.append(snap.to_json())

    # ---- Persist ----
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        if self._bias_jsonl:
            with open(os.path.join(out_dir, "bias_samples.jsonl"), "w") as f:
                for row in self._bias_jsonl:
                    f.write(json.dumps(row) + "\n")
        if self._fuse_jsonl:
            with open(os.path.join(out_dir, "fusion_snaps.jsonl"), "w") as f:
                for row in self._fuse_jsonl:
                    f.write(json.dumps(row) + "\n")
            # Also write a gzipped copy for large files
            try:
                import gzip
                with gzip.open(os.path.join(out_dir, "fusion_snaps.jsonl.gz"), "wt") as gf:
                    for row in self._fuse_jsonl:
                        gf.write(json.dumps(row) + "\n")
            except Exception:
                pass
