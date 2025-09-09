# swarm_ml/snapshots.py
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
    X: np.ndarray                      # (N_nodes, d) node features
    mus: np.ndarray                    # (N_nodes, 6) local posteriors mu_i
    Ps: np.ndarray                     # (N_nodes, 6, 6) local posteriors P_i
    gt_pos: np.ndarray                 # (3,)
    los: Optional[Dict[str, float]] = None
    exp: Optional[str] = None          # NEW: experiment id for split-aware training

    def to_json(self) -> Dict[str, Any]:
        return {
            "timestamp": float(self.timestamp),
            "order": list(self.order),
            "X": self.X.tolist(),
            "mus": self.mus.tolist(),
            "Ps": self.Ps.tolist(),
            "gt_pos": self.gt_pos.tolist(),
            "los": None if self.los is None else {k: float(v) for k, v in self.los.items()},
            "exp": None if self.exp is None else str(self.exp),
        }


class SnapshotCollector:
    """
    Collects:
      - BiasNet samples (features + bias labels)
      - FusionNet snaps (per-timestep node features + local posteriors + GT target position)
    Saves JSONL files under the experiment out directory.
    """
    def __init__(self,
                 exp_name: Optional[str],
                 query_timestamps: np.ndarray,
                 roles,
                 tag_map: Dict[str, List[int]],
                 gt_T_by_robot,
                 tag_moment_arms=None,
                 base_var=1.0,
                 pair_corr=0.0,
                 huber_delta=1.0,
                 los_verbose=False,
                 height_series: Optional[Dict[str, np.ndarray]] = None):
        self.ts = np.asarray(query_timestamps, dtype=float)
        self.roles = roles
        self.tag_map = tag_map
        self.gt_T = gt_T_by_robot
        # Stable tracker vocabulary for one-hot identity features
        self.tracker_vocab = sorted(list(roles.trackers))
        self.exp_name = exp_name
        self.tag_moment_arms = tag_moment_arms
        self.base_var = float(base_var)
        self.pair_corr = float(pair_corr)
        self.huber_delta = float(huber_delta)
        self.los_adapter = LOSAdapter(LOSConfig(verbose=los_verbose))
        self.height_series = height_series  # dict: robot -> np.ndarray aligned with self.ts

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
        Log a sample consistent with runtime geometry (eff_sensor_pos used).
        Supervision uses GT target position at i.
        """
        t = float(self.ts[i])
        tgt = self.roles.target
        p_tgt = self.gt_pos[tgt][i]
        true_range = float(np.linalg.norm(p_tgt - eff_sensor_pos_used))

        # Δz from PX4 height if available and requested
        tgt = self.roles.target
        h_trk = None; h_tgt = None
        if self.height_series is not None:
            try:
                h_trk = float(self.height_series.get(trk, [None] * len(self.ts))[i])
                h_tgt = float(self.height_series.get(tgt, [None] * len(self.ts))[i])
            except (IndexError, TypeError, ValueError):
                h_trk = None; h_tgt = None

        # Keep LOS neutral (0.5) if not provided
        feat = build_measurement_features(
            tracker_pos=np.asarray(eff_sensor_pos_used, dtype=float),
            target_pred_pos=None,               # training parity: no bearing/dist inputs
            uwb_range=float(z_agg),
            los_score=None if los_score is None else float(los_score),
            height_tracker=h_trk,
            height_target=h_tgt
        )
        feat = np.asarray(feat, dtype=float)

        # --- Pair-quality stats for BiasNet + identity one-hot ---
        from swarm_ml.tagmap import robust_range_aggregate
        # recompute pair variance/meta using the same settings as runtime
        _, R_pair, meta_pairs = robust_range_aggregate(
            pair_df, base_var=self.base_var, rho=self.pair_corr, huber_delta=self.huber_delta
        )
        m_eff = float(meta_pairs.get("m_eff", 1.0))
        # Prefer an IQR from the rows we have at this t (m=2 supported)
        if pair_df is not None and not pair_df.empty:
            zs = pair_df["range"].to_numpy(dtype=float)
            if zs.size >= 3:
                q25, q75 = np.percentile(zs, [25, 75]); iqr = float(max(0.0, q75 - q25))
            elif zs.size == 2:
                iqr = float(abs(zs[1] - zs[0]))
            else:
                iqr = 0.0
        else:
            iqr = 0.0
        # one-hot identity for tracker
        onehot = np.zeros(len(self.tracker_vocab), dtype=float)
        if trk in self.tracker_vocab:
            onehot[self.tracker_vocab.index(trk)] = 1.0
        feat = np.hstack([feat, [float(R_pair), m_eff, iqr], onehot])

        self._bias_jsonl.append({
            "features": feat.tolist(),
            "bias": float(z_agg - true_range),
            "meta": {
                "exp": None if self.exp_name is None else str(self.exp_name),
                "timestamp": t,
                "tracker": trk,
                "target": tgt,
                "z_agg": float(z_agg),
                "true_range": true_range,
                "los": None if los_score is None else float(los_score),
                "R_pair": float(R_pair)
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

        snap = FusionSnap(
            timestamp=t, order=order, X=X, mus=mus, Ps=Ps, gt_pos=gt_pos,
            los=None, exp=self.exp_name
        )
        self._fuse_jsonl.append(snap.to_json())

        # ---- Persist ----
    def save(self, out_dir: str) -> None:
        os.makedirs(out_dir, exist_ok=True)
        bias_path = os.path.join(out_dir, "bias_samples.jsonl")
        fuse_path = os.path.join(out_dir, "fusion_snaps.jsonl")
        fuse_gz_path = fuse_path + ".gz"

        # Always create files, even if empty, so downstream CLIs never 404.
        with open(bias_path, "w") as f:
            for row in self._bias_jsonl:
                f.write(json.dumps(row) + "\n")

        with open(fuse_path, "w") as f:
            for row in self._fuse_jsonl:
                f.write(json.dumps(row) + "\n")

        # Also write a gz copy of fusion snaps; ignore gzip errors but be verbose
        try:
            import gzip
            with gzip.open(fuse_gz_path, "wt") as gf:
                for row in self._fuse_jsonl:
                    gf.write(json.dumps(row) + "\n")
        except Exception as e:
            print(f"[COLLECT] Note: could not write {fuse_gz_path}: {e}")

        # Quick stats so you can validate immediately
        stats = {
            "n_bias_samples": int(len(self._bias_jsonl)),
            "n_fusion_snaps": int(len(self._fuse_jsonl)),
            "bias_path": bias_path,
            "fusion_path": fuse_path,
            "fusion_path_gz": fuse_gz_path,
        }
        with open(os.path.join(out_dir, "collect_stats.json"), "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[COLLECT] bias samples: {stats['n_bias_samples']:,} -> {bias_path}")
        print(f"[COLLECT] fusion snaps: {stats['n_fusion_snaps']:,} -> {fuse_path} (+ .gz)")

        # Persist a simple schema for node feature layout to aid training
        schema = {
            "node_features_layout": {
                "0": "var_pos_trace",
                "1": "reliability", 
                "2": "z_agg_center",
                "3": "R_eff",
                "4": "geom_ez_abs",
                "5": "los_score",
                "6": "gate_sigma",
                "7": "nis_ema",
                "8": "R_pair",
                "9": "m_eff",
                "10": "iqr"
            },
            "description": "Per-node feature indices used for FusionNet training (enhanced with tag-pair stats)",
        }
        with open(os.path.join(out_dir, "fusion_schema.json"), "w") as f:
            json.dump(schema, f, indent=2)
