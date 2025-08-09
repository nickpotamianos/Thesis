# sota/miluv/loader.py
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple
import numpy as np
import pandas as pd
import math
import os

# This is the devkit loader you already have
from miluv.data import DataLoader as MiluvDataLoader

# ---------- Small event containers ----------

@dataclass
class ImuEvent:
    t: float
    robot: int         # 0,1,2
    source: str        # "px4" or "cam"
    omega: np.ndarray  # rad/s, shape (3,)
    accel: np.ndarray  # m/s^2, shape (3,)

@dataclass
class UwbEvent:
    t: float
    from_id: int
    to_id: int
    rng: float
    std: float
    is_intertag: bool  # True if tag<->tag (10/11/20/21/30/31), False if anchor<->tag

@dataclass
class HeightEvent:
    t: float
    robot: int
    z_range: float

# ---------- Adapter ----------

TAG_IDS = {10,11,20,21,30,31}         # From MILUV, tags on robots (paper §5.2.1)
ANCHOR_IDS = {0,1,2,3,4,5}            # Six anchors (paper §4.2 + Fig. 4)

def _robot_index_for_tag(tag_id: int) -> Optional[int]:
    # 10/11 -> robot 0, 20/21 -> robot 1, 30/31 -> robot 2
    if tag_id in (10,11): return 0
    if tag_id in (20,21): return 1
    if tag_id in (30,31): return 2
    return None

class MiluvAdapter:
    """
    Thin wrapper around the MILUV devkit DataLoader that:
      - exposes anchors + tag moment arms,
      - merges per-robot CSVs into a single time-ordered stream of events,
      - applies basic dedup/gating for UWB,
      - (optionally) removes IMU biases using the devkit bias estimates,
      - returns height events with bias already corrected (devkit handles this).
    """

    def __init__(
        self,
        exp_name: str,
        exp_dir: str = "./data",
        imu_source: str = "px4",          # "px4" | "cam" | "both"
        include_height: bool = True,
        include_passive: bool = False,    # currently unused; we can add passive later
        include_cir: bool = False,        # currently unused in event stream
        remove_imu_bias: bool = True,
        uwb_sigma_default: float = 0.35,  # used if 'std' not in csv
        uwb_min: float = 0.05,
        uwb_max: float = 20.0,
        rr_dedup_dt: float = 0.02,        # dedup inter-tag ranges by time
    ):
        self.exp_name = exp_name
        self.exp_dir = exp_dir
        self.imu_source = imu_source
        self.include_height = include_height
        self.include_passive = include_passive
        self.include_cir = include_cir
        self.remove_imu_bias = remove_imu_bias
        self.uwb_sigma_default = float(uwb_sigma_default)
        self.uwb_min = float(uwb_min)
        self.uwb_max = float(uwb_max)
        self.rr_dedup_dt = float(rr_dedup_dt)

        # Ask the devkit to read/prepare everything for us.
        # DataLoader consults config/experiments.csv to know num_robots & anchor constellation (paper §7.2)
        self.mv = MiluvDataLoader(
            exp_name,
            exp_dir=exp_dir,
            imu=("both" if imu_source == "both" else imu_source),
            uwb=True,
            height=include_height,
            mag=False,
            cir=include_cir,
            barometer=False,
            remove_imu_bias=remove_imu_bias,
        )

        # Anchors (positions in the Vicon/world frame Fa) and tag moment arms (r^ji_i),
        # both provided by the devkit configs (paper §4.2 and §5.2.1)
        self.anchors: Dict[int, np.ndarray] = self.mv.anchors  # {0..5: (x,y,z)}
        self.tag_moment_arms: Dict[str, Dict[int, List[float]]] = self.mv.tag_moment_arms  # {"ifo00i": {10:[...],11:[...]},...}

        # Robots present for this experiment, in devkit ID form ("ifo001"...)
        self.robot_ids: List[str] = list(self.mv.data.keys())         # e.g., ["ifo001","ifo002","ifo003"]
        self.robot_count = len(self.robot_ids)

        # Assemble event tables (pandas) now; stream them later.
        self._imu_tables = self._prepare_imu_tables()
        self._uwb_table  = self._prepare_uwb_table()
        self._height_table = self._prepare_height_table() if self.include_height else None

    # ---------- public API ----------

    def get_static(self) -> Dict:
        """Static metadata needed by downstream code."""
        # Convert tag offsets to a flat {tag_id: np.array([x,y,z])} for convenience
        tag_offsets: Dict[int, np.ndarray] = {}
        for ridx, rid in enumerate(self.robot_ids):
            for tag_id, vec in self.tag_moment_arms[rid].items():
                tag_offsets[int(tag_id)] = np.array(vec, dtype=float)
        return {
            "anchors": self.anchors,            # {0..5: np.array([x,y,z])}
            "tag_offsets": tag_offsets,         # {10,11,20,21,30,31: np.array([x,y,z])}
            "robot_ids": self.robot_ids,        # ["ifo001", ...]
            "robot_count": self.robot_count
        }

    def iter_events(self, t0: float = 0.0, t1: Optional[float] = None) -> Iterator[Tuple[str, object]]:
        """
        Merge IMU, UWB (and height) into a single time-ordered stream.
        Yields tuples: ("IMU", ImuEvent) | ("UWB", UwbEvent) | ("HEIGHT", HeightEvent)
        """
        # Build three cursors and walk them in time order
        sources: List[pd.DataFrame] = []
        if self._imu_tables is not None:
            sources.append(self._imu_tables)
        if self._uwb_table is not None:
            sources.append(self._uwb_table)
        if self._height_table is not None:
            sources.append(self._height_table)

        if not sources:
            return iter(())

        # Concatenate and sort once — simpler and fast enough
        big = pd.concat(sources, ignore_index=True).sort_values("timestamp")
        if t1 is not None:
            big = big[(big["timestamp"] >= t0) & (big["timestamp"] <= t1)]
        else:
            big = big[big["timestamp"] >= t0]

        # Emit unified packets
        for _, row in big.iterrows():
            typ = row["_kind"]
            ts  = float(row["timestamp"])
            if typ == "IMU":
                ev = ImuEvent(
                    t=ts,
                    robot=int(row["_robot"]),
                    source=row["_src"],
                    omega=np.array([row["gyr_x"], row["gyr_y"], row["gyr_z"]], dtype=float),
                    accel=np.array([row["acc_x"], row["acc_y"], row["acc_z"]], dtype=float),
                )
                yield ("IMU", ev)

            elif typ == "UWB":
                ev = UwbEvent(
                    t=ts,
                    from_id=int(row["from_id"]),
                    to_id=int(row["to_id"]),
                    rng=float(row["range"]),
                    std=float(row["std"]),
                    is_intertag=bool(row["_is_intertag"]),
                )
                yield ("UWB", ev)

            elif typ == "HEIGHT":
                ev = HeightEvent(
                    t=ts, robot=int(row["_robot"]), z_range=float(row["range"])
                )
                yield ("HEIGHT", ev)

    # ---------- prep helpers ----------

    def _prepare_imu_tables(self) -> Optional[pd.DataFrame]:
        rows = []
        use_px4 = self.imu_source in ("px4", "both")
        use_cam = self.imu_source in ("cam", "both")

        for ridx, rid in enumerate(self.robot_ids):
            if use_px4 and "imu_px4" in self.mv.data[rid]:
                df = self.mv.data[rid]["imu_px4"]
                # devkit columns: angular_velocity.{x,y,z}, linear_acceleration.{x,y,z} (paper Table 7)
                rows.append(pd.DataFrame({
                    "timestamp": df["timestamp"].astype(float),
                    "gyr_x": df["angular_velocity.x"].astype(float),
                    "gyr_y": df["angular_velocity.y"].astype(float),
                    "gyr_z": df["angular_velocity.z"].astype(float),
                    "acc_x": df["linear_acceleration.x"].astype(float),
                    "acc_y": df["linear_acceleration.y"].astype(float),
                    "acc_z": df["linear_acceleration.z"].astype(float),
                    "_kind": "IMU",
                    "_robot": ridx,
                    "_src": "px4",
                }))
            if use_cam and "imu_cam" in self.mv.data[rid]:
                df = self.mv.data[rid]["imu_cam"]
                rows.append(pd.DataFrame({
                    "timestamp": df["timestamp"].astype(float),
                    "gyr_x": df["angular_velocity.x"].astype(float),
                    "gyr_y": df["angular_velocity.y"].astype(float),
                    "gyr_z": df["angular_velocity.z"].astype(float),
                    "acc_x": df["linear_acceleration.x"].astype(float),
                    "acc_y": df["linear_acceleration.y"].astype(float),
                    "acc_z": df["linear_acceleration.z"].astype(float),
                    "_kind": "IMU",
                    "_robot": ridx,
                    "_src": "cam",
                }))

        if not rows:
            return None
        return pd.concat(rows, ignore_index=True)

    def _prepare_height_table(self) -> Optional[pd.DataFrame]:
        if not self.include_height:
            return None
        rows = []
        for ridx, rid in enumerate(self.robot_ids):
            if "height" not in self.mv.data[rid]:
                continue
            df = self.mv.data[rid]["height"]
            # devkit already subtracts the bias using config/height/bias.yaml (paper §10)
            tmp = pd.DataFrame({
                "timestamp": df["timestamp"].astype(float),
                "range": df["range"].astype(float),
                "_kind": "HEIGHT",
                "_robot": ridx
            })
            rows.append(tmp)
        if not rows:
            return None
        return pd.concat(rows, ignore_index=True)

    def _prepare_uwb_table(self) -> Optional[pd.DataFrame]:
        # Each robot logs a copy of UWB ranges; we merge & dedup on (timestamp, from_id, to_id)
        uwb_frames = []
        for rid in self.robot_ids:
            if "uwb_range" not in self.mv.data[rid]:
                continue
            df = self.mv.data[rid]["uwb_range"]
            keep_cols = [
                "timestamp","from_id","to_id","range","std"
            ]
            present = [c for c in keep_cols if c in df.columns]
            tmp = df[present].copy()
            uwb_frames.append(tmp)

        if not uwb_frames:
            return None

        U = pd.concat(uwb_frames, ignore_index=True)
        # Dedup identical TWR rows recorded by multiple robots
        U.sort_values(["timestamp","from_id","to_id"], inplace=True)
        U = U.drop_duplicates(subset=["timestamp","from_id","to_id"], keep="first")

        # Fill/clean STD and filter by range limits
        if "std" not in U.columns:
            U["std"] = self.uwb_sigma_default
        U["std"] = U["std"].astype(float).abs().clip(lower=1e-6, upper=5.0)

        # Gate ranges
        U = U[(U["range"] >= self.uwb_min) & (U["range"] <= self.uwb_max)]

        # Mark inter-tag vs anchor<->tag
        U["_is_intertag"] = (U["from_id"].isin(list(TAG_IDS)) & U["to_id"].isin(list(TAG_IDS))).astype(bool)

        # Dedup *inter-tag* factors by a small dt to avoid pounding iSAM2 (matches your CLI flag)
        last_time_by_pair: Dict[Tuple[int,int], float] = {}
        mask_keep = np.ones(len(U), dtype=bool)
        for idx, row in U.iterrows():
            if row["_is_intertag"]:
                a, b = int(row["from_id"]), int(row["to_id"])
                key = (min(a,b), max(a,b))
                last = last_time_by_pair.get(key, -1e9)
                if (float(row["timestamp"]) - last) < self.rr_dedup_dt:
                    mask_keep[idx] = False
                else:
                    last_time_by_pair[key] = float(row["timestamp"])
        U = U[mask_keep]

        U["_kind"] = "UWB"
        return U.reset_index(drop=True)
