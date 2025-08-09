# sota/factor_graph/fg_runner.py
import time, yaml, gtsam, numpy as np
import pandas as pd
import os
from pathlib import Path
from miluv.data import DataLoader
from gtsam import (
    PreintegratedImuMeasurements, ImuFactor, CombinedImuFactor,
    imuBias, noiseModel
)
from sota.uwb.infer_uwb          import UwbML
from sota.factor_graph.range_factory_std import RangeFactoryStd
from sota.factor_graph.utils_io  import ensure_dir, save_estimates, save_graph_values   ### NEW

try:
    from sota.miluv.loader import MiluvAdapter
except Exception:
    MiluvAdapter = None

# put near the top of fg_runner.py
def frd_to_flu(v):
    """PX4 FRD -> FLU (keep x, flip y and z). Works for accel and gyro."""
    return np.array([v[0], -v[1], -v[2]], dtype=float)

def _tag_to_robot(tag_id: int):
    """Map 10/20/30 → robot 0/1/2. Return None if not a robot tag."""
    base = (tag_id // 10) * 10
    return {10: 0, 20: 1, 30: 2}.get(base, None)

def _get_anchor_pos(anchors_dict, anchor_id):
    """
    Return a (3,) float64 numpy array or None.
    Handles all known MILUV layouts.
    """
    if anchors_dict is None:
        return None

    cand = None
    # flat layout {0: xyz, 1: …}
    if anchor_id in anchors_dict:
        cand = anchors_dict[anchor_id]
    else:
        # nested layout {'default': {0: xyz, …}}
        for v in anchors_dict.values():
            if isinstance(v, dict) and anchor_id in v:
                cand = v[anchor_id]
                break

    if cand is None:
        return None

    # ---- fix shape + dtype here --------------------------------------
    a = np.asarray(cand, dtype=float).reshape(-1)   # flatten ((1,3) → (3,))
    if a.size != 3:
        raise ValueError(f"Anchor {anchor_id} has invalid size {a.shape}")
    return a

# ----------------------------------------------------------------------
def fuse_range_and_cir(robot_table, max_dt=0.15, return_unfused=False):
    """
    Merge UWB range with closest CIR within ±max_dt (s), direction-aware.
    If return_unfused=True, also return the range rows that had no CIR match.
    """
    rng = robot_table["uwb_range"][["timestamp",
                                    "from_id", "to_id",
                                    "range", "gt_range"]].copy()
    cir = robot_table["uwb_cir"  ][["timestamp",
                                    "from_id", "to_id",
                                    "cir"]].copy()

    rng = rng.sort_values("timestamp")
    cir = cir.sort_values("timestamp")

    merged = pd.merge_asof(
        rng, cir,
        on       = "timestamp",
        by       = ["from_id", "to_id"],
        tolerance= max_dt,
        direction= "nearest"
    )

    fused = merged.dropna(subset=["cir"])
    if not return_unfused:
        return fused

    rng_only = merged[merged["cir"].isna()].drop(columns=["cir"])
    return fused, rng_only

POSE_KEY   = lambda r, k: gtsam.symbol(chr(ord('P') + r), k)   # P,Q,R
VEL_KEY    = lambda r, k: gtsam.symbol(chr(ord('V') + r), k)   # V,W,X
BIAS_KEY   = lambda r, k: gtsam.symbol(chr(ord('B') + r), k)   # B,C,D

class FGCoordinator:
    def __init__(self, exp, save_dir=None, cir_dt=0.15, uwb_sigma=0.25, uwb_min=0.05, uwb_max=25.0, rr_dedup_dt=0.02):                    ### NEW
        print(f"=== INITIALIZING FGCoordinator for experiment: {exp} ===")
        self.exp  = exp
        self.save_dir = save_dir                               ### NEW
        self.cir_dt = cir_dt                                   ### NEW
        self.uwb_sigma = uwb_sigma                             ### NEW
        self.uwb_min = uwb_min                                 ### NEW
        self.uwb_max = uwb_max                                 ### NEW
        self.rr_dedup_dt = rr_dedup_dt                         ### NEW
        self.packet_counter = 0                                ### NEW
        self.mv   = DataLoader(exp, exp_dir="data/three_robots", cir=True, height=False)   # height unused for now
        print(f"DataLoader created successfully")
        print(f"Available robots: {list(self.mv.data.keys())}")
        print(f"Using CIR tolerance: {self.cir_dt}s, UWB sigma: {self.uwb_sigma}m")
        
        # flatten all anchor vectors in place
        if self.mv.anchors:
            print(f"Raw anchors before flattening: {self.mv.anchors}")
            for k, v in self.mv.anchors.items():
                old_shape = np.asarray(v).shape
                self.mv.anchors[k] = np.asarray(v, dtype=float).reshape(-1)
                print(f"  Anchor {k}: {old_shape} -> {self.mv.anchors[k].shape}, value: {self.mv.anchors[k]}")
        else:
            print("No anchors found in dataset")
                
        # IMU pre‑integration parameters (calibrated once)
        print("=== SETTING UP IMU PRE-INTEGRATION ===")
        # --- IMU preintegration params (Combined) ---
        g = 9.81
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(g)  # gravity magnitude; -Z in world

        # Not all builds have setGravityVector; MakeSharedU already sets gravity.
        if hasattr(params, "setGravityVector"):
            params.setGravityVector(np.array([0.0, 0.0, -g], dtype=float))

        # Sensor noise (tune to your IMU)
        params.setAccelerometerCovariance(np.eye(3) * (0.03**2))   # (m/s^2)^2
        params.setGyroscopeCovariance(np.eye(3) * (0.002**2))      # (rad/s)^2
        params.setIntegrationCovariance(np.eye(3) * ((1e-3)**2))   # be explicit with parentheses

        # Bias random walk (some older wrappers miss one or more; guard them)
        if hasattr(params, "setBiasAccCovariance"):
            params.setBiasAccCovariance(np.eye(3) * 1e-4)
        if hasattr(params, "setBiasOmegaCovariance"):
            params.setBiasOmegaCovariance(np.eye(3) * 1e-6)
        if hasattr(params, "setBiasAccOmegaIntCovariance"):
            params.setBiasAccOmegaIntCovariance(np.eye(6) * 1e-5)

        print(f"Combined IMU params created with gravity: {g}")
        print(f"Gyro covariance: {0.002**2}")
        print(f"Accel covariance: {0.03**2}")

        # Initializers per robot
        self.preint = {}
        self.prev_imu_t = {}
        self.body_is_frd = {0: True, 1: True, 2: True}
        for r in range(3):
            self.preint[r] = gtsam.PreintegratedCombinedMeasurements(params, imuBias.ConstantBias())
            self.prev_imu_t[r] = None
            print(f"  Robot {r}: Combined IMU preintegrator initialized")
        
        print("=== SETTING UP TIME-BASED KEY TRACKING ===")
        from bisect import bisect_right
        
        # Make bisect_right available to other methods
        import bisect
        self.bisect_right = bisect.bisect_right
        
        # --- per-robot key-time index ---
        self.key_times = {r: [] for r in range(3)}   # monotonically increasing times
        self.key_ids   = {r: [] for r in range(3)}   # matching pose index numbers

        # seed with the first IMU time per robot (key 0)
        for ridx, (name, rob) in enumerate(self.mv.data.items()):
            imu_df = rob["imu_px4"].sort_values("timestamp")
            if imu_df.empty:
                raise RuntimeError(f"Robot {ridx} has no IMU data; cannot seed key timeline.")
            t0 = float(imu_df["timestamp"].iloc[0])
            self.key_times[ridx].append(t0)
            self.key_ids[ridx].append(0)
            print(f"  Robot {ridx} ({name}): seeded key timeline with t0={t0:.3f}s")

        print("=== SETTING UP UWB ML FRONTEND ===")
        self.rng_factory = RangeFactoryStd(UwbML(exp))          # pre‑trained CNNs
        print(f"RangeFactoryStd created for experiment: {exp}")
        
        # RR dedup within small dt
        self.rr_last_seen = {}   # dict[(key_low, key_high)] = last_time
        print(f"Robot-to-robot deduplication window: {self.rr_dedup_dt}s")
        print(f"UWB range gating: [{self.uwb_min}, {self.uwb_max}] meters")
        
        # Debug: check what anchors are loaded
        print("=== ANCHOR SUMMARY ===")
        print("Anchors loaded:", sorted(self.mv.anchors.keys()) if self.mv.anchors else "None")
        if self.mv.anchors:
            for k, v in self.mv.anchors.items():
                print(f"  Anchor {k}: {v} (type: {type(v)}, shape: {v.shape})")
        
        print("=== SETTING UP FACTOR GRAPH ===")
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        # Create ISAM2 with more conservative parameters for numerical stability
        params = gtsam.ISAM2Params()
        if hasattr(gtsam.ISAM2Params, 'QR'):
            params.setFactorization(gtsam.ISAM2Params.QR)
        if hasattr(params, 'setRelinearizeThreshold'):
            params.setRelinearizeThreshold(0.01)   # smaller threshold for more frequent relinearization
        elif hasattr(params, 'relinearizeThreshold'):
            params.relinearizeThreshold = 0.01
        if hasattr(params, 'setRelinearizeSkip'):
            params.setRelinearizeSkip(1)           # relinearize every update
        elif hasattr(params, 'relinearizeSkip'):
            params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)

        # Load anchors into the factory (creates Point3 variables with tight priors)
        if self.mv.anchors:
            print("Loading anchors into RangeFactoryStd...")
            # Convert anchors to the required format: dict[int, np.ndarray]
            anchors_xyz = {}
            for aid, pos in self.mv.anchors.items():
                anchors_xyz[aid] = np.asarray(pos, dtype=float).reshape(3)
            
            self.rng_factory.load_anchors(
                self.graph,      # NonlinearFactorGraph
                self.values,     # gtsam.Values (initial values container)
                anchors_xyz=anchors_xyz
            )
            print(f"Loaded {len(anchors_xyz)} anchors as Point3 variables with tight priors")
        else:
            print("No anchors found - skipping anchor setup")
        print("GTSAM objects created: NonlinearFactorGraph, Values, ISAM2")

        # --- priors -------------------------------------------------------
        print("=== ADDING POSE PRIORS ===")
        
        def pose_prior_noise(t_sigma=0.5, rp_sigma=0.05, yaw_sigma=np.pi/3):
            """Create pose prior noise with different sigmas for translation and rotation."""
            # [roll, pitch, yaw, x, y, z]
            sig = np.array([rp_sigma, rp_sigma, yaw_sigma, t_sigma, t_sigma, t_sigma], float)
            return gtsam.noiseModel.Diagonal.Sigmas(sig)
        
        k0 = 0
        for ridx, robot in enumerate(self.mv.data.keys()):
            k = POSE_KEY(ridx, k0)
            R0 = self._estimate_initial_orientation(ridx)    # <- NEW
            P0 = gtsam.Pose3(R0, gtsam.Point3(0, 0, 0))
            
            # Robot 0 gets tight prior, others get looser priors to avoid conflicts with ranges
            if ridx == 0:
                prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]*6))  # tight
                print(f"  Robot {ridx} ({robot}): TIGHT pose prior, key = {k}")
            else:
                prior_noise = pose_prior_noise(t_sigma=1.0, rp_sigma=0.05, yaw_sigma=np.pi/3)  # looser
                print(f"  Robot {ridx} ({robot}): SOFT pose prior, key = {k}")
            
            self.values.insert(k, P0)
            self.graph.add(gtsam.PriorFactorPose3(k, P0, prior_noise))
            print(f"    Added pose prior factor for robot {ridx}")

        # velocity and bias priors
        print("=== ADDING VELOCITY AND BIAS PRIORS ===")
        for ridx in range(3):
            vel_key = VEL_KEY(ridx,0)
            bias_key = BIAS_KEY(ridx,0)
            print(f"  Robot {ridx}: vel_key = {vel_key}, bias_key = {bias_key}")
            
            self.values.insert( vel_key, np.zeros(3) )
            self.values.insert( bias_key, gtsam.imuBias.ConstantBias() )
            
            v_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
            b_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            
            self.graph.add(gtsam.PriorFactorVector( vel_key, np.zeros(3), v_noise))
            self.graph.add(gtsam.PriorFactorConstantBias( bias_key,
                                                          gtsam.imuBias.ConstantBias(),
                                                          b_noise))
            print(f"    Added velocity and bias priors for robot {ridx}")

        print(f"Initial graph size: {self.graph.size()} factors")
        print(f"Initial values size: {self.values.size()} variables")

        # bookkeeping
        self.last_key = {r: k0 for r in range(3)}
        print(f"Initial last_key mapping: {self.last_key}")
        
        print("=== BUILDING TIMELINE ===")
        self.iterator   = self._make_timeline()
        print("Timeline iterator created")
        
        print("=== BOOTSTRAP: committing priors & anchors to ISAM2 ===")
        self._flush_graph()           # pushes pose/vel/bias priors and anchor priors
        print("=== BOOTSTRAP DONE ===")
        print("=== INITIALIZATION COMPLETE ===\n")

    def _estimate_initial_orientation(self, ridx: int, seconds: float = 0.5) -> gtsam.Rot3:
        """
        Estimate roll/pitch from the average accelerometer (gravity) of the first
        'seconds' seconds. Assumes IMU is FRD; we convert to FLU first.
        """
        name = list(self.mv.data.keys())[ridx]
        imu_df = self.mv.data[name]["imu_px4"].sort_values("timestamp")
        t0 = float(imu_df["timestamp"].iloc[0])
        window = imu_df[imu_df["timestamp"] <= t0 + seconds]

        ax = window["linear_acceleration.x"].to_numpy().mean()
        ay = window["linear_acceleration.y"].to_numpy().mean()
        az = window["linear_acceleration.z"].to_numpy().mean()
        a_raw = np.array([ax, ay, az], dtype=float)

        # Convert PX4 FRD -> FLU
        a = self._maybe_convert(a_raw, is_gyro=False, robot=ridx)

        # FLU at rest: a ≈ [0, 0, -g]
        roll  = np.arctan2(a[1], -a[2])                           # <-- sign fix
        pitch = np.arctan2(a[0],  np.sqrt(a[1]**2 + a[2]**2))     # <-- sign fix
        yaw   = 0.0
        return gtsam.Rot3.Ypr(yaw, pitch, roll)  # GTSAM uses Y,P,R order here

    def _maybe_convert(self, v, is_gyro=False, robot=0):
        if self.body_is_frd[robot]:
            return frd_to_flu(v)  # x, -y, -z
        return v

    def _key_for_time(self, robot: int, t: float) -> int:
        """
        Return the pose Key for 'robot' that is current at time t:
        the last key whose birth time <= t. Falls back to last_key if needed.
        """
        ts = self.key_times[robot]
        ks = self.key_ids[robot]
        if not ts:
            # shouldn't happen, but be defensive
            return POSE_KEY(robot, self.last_key[robot])
        idx = self.bisect_right(ts, t) - 1
        if idx < 0:  # measurement before first key time
            idx = 0
        return POSE_KEY(robot, ks[idx])

    def _isam_has(self, key):
        """Check if ISAM2 already contains a variable with this key."""
        try:
            return self.isam.calculateEstimate().exists(key)
        except Exception:
            return False

    def _keys_in_factors(self, graph):
        """Extract all keys referenced by factors in the graph."""
        ks = set()
        for i in range(graph.size()):
            f = graph.at(i)
            klist = f.keys()
            # Handle both KeyVector and list types
            if hasattr(klist, 'size'):
                for j in range(klist.size()):
                    ks.add(int(klist.at(j)))
            else:
                for key in klist:
                    ks.add(int(key))
        return ks

    def _keys_in_values(self, values):
        """Extract all keys present in the values container."""
        ks = set()
        it = values.keys()
        # Handle KeyVector iteration
        try:
            for j in range(len(it)):
                ks.add(int(it[j]))
        except:
            # Fallback for different GTSAM versions
            for key in it:
                ks.add(int(key))
        return ks

    def _flush_graph(self):
        """
        Push the factors accumulated in self.graph/self.values into ISAM2,
        then clear the buffers so the next window starts empty.
        """
        if self.graph.size() == 0:
            return                                             # nothing to do

        print(f"    Flushing graph with {self.graph.size()} factors, {self.values.size()} values to ISAM2")
        
        # Connectivity check: verify all factors reference known variables
        ks_graph = self._keys_in_factors(self.graph)
        ks_values = self._keys_in_values(self.values)
        try:
            ks_isam = set(int(k) for k in self.isam.calculateEstimate().keys())
        except Exception:
            ks_isam = set()

        unknown = ks_graph - (ks_isam | ks_values)
        if unknown:
            print("WARNING: factors reference keys unknown to ISAM and not in this batch:", unknown)
        
        # 1) update ISAM2
        self.isam.update(self.graph, self.values)
        print(f"    ISAM2 update successful")

        # 2) optional on‑disk dump
        if self.save_dir:
            # Use a more meaningful stamp based on current estimates count
            stamp = f"{self.isam.getFactorsUnsafe().size():06d}"
            save_graph_values(self.graph, self.values,
                             f"{self.save_dir}/graph_{stamp}.graph",
                             f"{self.save_dir}/values_{stamp}.values")
            print(f"    Saved graph/values to disk with stamp {stamp}")

        # 3) ***important*** – keep the ISAM2 content but start a fresh window
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        print(f"    Graph and values cleared for next window")

    # merge all robot‑sensor streams chronologically -----------------------
    def _make_timeline(self):
        print("  Building timeline from all robot sensor data...")
        rows = []
        total_imu_packets = 0
        total_uwb_packets = 0

        for ridx, (_rname, rob) in enumerate(self.mv.data.items()):
            print(f"    Processing robot {ridx} ({_rname})...")

            # ----------  IMU  ------------------------------------------------
            imu = rob["imu_px4"][[
                "timestamp",
                "angular_velocity.x", "angular_velocity.y", "angular_velocity.z",
                "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"
            ]].copy()
            imu["sensor"] = "imu"
            imu["robot"]  = ridx
            print(f"      IMU packets: {len(imu)}")
            total_imu_packets += len(imu)
            rows.append(imu)

            # ----------  UWB  (range + CIR)  ---------------------------------
            print(f"      Fusing UWB range and CIR data...")
            fused, rng_only = fuse_range_and_cir(rob, max_dt=self.cir_dt, return_unfused=True)

            uwb = fused.copy()
            uwb["sensor"] = "uwb"                # fused with CIR
            uwb["robot"]  = ridx
            print(f"      UWB fused (with CIR): {len(uwb)}")
            rows.append(uwb)

            uwb_ro = rng_only.copy()
            uwb_ro["sensor"] = "uwb_range_only"  # range-only fallback
            uwb_ro["robot"]  = ridx
            print(f"      UWB range-only (no CIR): {len(uwb_ro)}")
            rows.append(uwb_ro)

            total_uwb_packets += len(uwb) + len(uwb_ro)

        print(f"  Timeline summary:")
        print(f"    Total IMU packets: {total_imu_packets}")
        print(f"    Total UWB packets: {total_uwb_packets}")
        
        # flatten list‑of‑DFs → list‑of‑dicts → sort by timestamp
        print("  Flattening and sorting timeline by timestamp...")
        all_dicts = (row for df in rows for row in df.to_dict("records"))
        sorted_timeline = sorted(all_dicts, key=lambda d: d["timestamp"])
        print(f"  Total timeline packets: {len(sorted_timeline)}")
        
        return iter(sorted_timeline)

    # ---------------------------------------------------------------------
    def run(self, max_sec=30):
        print(f"\n=== STARTING FACTOR GRAPH EXECUTION (max {max_sec} seconds) ===")
        t0 = time.time()
        packet_count = 0
        imu_count = 0
        uwb_count = 0
        
        for pkt in self.iterator:
            packet_count += 1
            elapsed = time.time() - t0
            
            if packet_count % 100 == 0:  # Print every 100 packets
                print(f"  Processed {packet_count} packets in {elapsed:.2f}s (IMU: {imu_count}, UWB: {uwb_count})")
                print(f"    Current graph size: {self.graph.size()} factors")
                print(f"    Current values size: {self.values.size()} variables")
            
            if pkt["sensor"] == "imu":
                imu_count += 1
                if imu_count <= 5:  # Detailed debug for first few IMU packets
                    print(f"    IMU packet {imu_count}: robot={pkt['robot']}, t={pkt['timestamp']:.6f}")
                self._handle_imu(pkt)
            elif pkt["sensor"] in ["uwb", "uwb_range_only"]:
                uwb_count += 1
                if uwb_count <= 5:  # Detailed debug for first few UWB packets
                    print(f"    UWB packet {uwb_count}: robot={pkt['robot']}, from={pkt['from_id']}, to={pkt['to_id']}, range={pkt['range']:.3f}, type={pkt['sensor']}")
                try:
                    self._handle_uwb(pkt)
                except Exception as e:
                    print(f"    ERROR in _handle_uwb: {e}")
                    print(f"    Packet details: {pkt}")
                    raise
            else:
                print(f"    Unknown sensor type: {pkt['sensor']}, skipping")
            
            # incremental optimisation - removed from here, now done in _flush_graph()
            
            if time.time() - t0 > max_sec:
                print(f"  Time limit reached after {packet_count} packets")
                break
        
        # Final flush to ensure all factors are in ISAM2
        self._flush_graph()
        
        print(f"=== EXECUTION COMPLETE ===")
        print(f"  Total packets processed: {packet_count} (IMU: {imu_count}, UWB: {uwb_count})")
        print(f"  Final execution time: {time.time() - t0:.2f} seconds")
        print(f"  Final ISAM2 estimates: {self.isam.calculateEstimate().size()} variables")
        print("Finished factor‑graph demo")

    # accumulate IMU; add IMU factor every ~2 seconds ---------------
    def _handle_imu(self, pkt):
        r = pkt["robot"]
        
        # integrate one raw measurement into the running pre‑integrator
        if self.prev_imu_t[r] is not None:
            dt = pkt["timestamp"] - self.prev_imu_t[r]
            accel_raw = np.array([pkt["linear_acceleration.x"],
                                  pkt["linear_acceleration.y"],
                                  pkt["linear_acceleration.z"]], dtype=float)
            gyro_raw  = np.array([pkt["angular_velocity.x"],
                                  pkt["angular_velocity.y"],
                                  pkt["angular_velocity.z"]], dtype=float)

            accel = self._maybe_convert(accel_raw, is_gyro=False, robot=r)
            gyro  = self._maybe_convert(gyro_raw, is_gyro=True, robot=r)
            
            # Specific force expected by GTSAM preintegration:
            # at rest, accel (body FLU) should be ≈ [0, 0, +g]
            accel = -accel
            
            if r == 0 and self.preint[r].deltaTij() < 0.1:  # Debug first robot, first few measurements
                print(f"      IMU integration robot {r}: dt={dt:.6f}, accel_raw={accel_raw}, accel_final={accel}")
            
            self.preint[r].integrateMeasurement(accel, gyro, dt)
        else:
            print(f"      First IMU measurement for robot {r} at t={pkt['timestamp']:.6f}")
            
        self.prev_imu_t[r] = pkt["timestamp"]

        # every 100 samples (≈0.75 s at 100 Hz) create a factor
        delta_t = self.preint[r].deltaTij()
        if delta_t < 0.75:    # ~0.75 s windows for better linearization
            return

        print(f"    Creating IMU factor for robot {r} after {delta_t:.3f} seconds of integration")
        
        k_prev = self.last_key[r]
        k_new  = k_prev + 1
        
        pose_key_prev = POSE_KEY(r,k_prev)
        vel_key_prev = VEL_KEY(r,k_prev)
        bias_key_prev = BIAS_KEY(r,k_prev)
        pose_key_new = POSE_KEY(r,k_new)
        vel_key_new = VEL_KEY(r,k_new)
        bias_key_new = BIAS_KEY(r,k_new)
        
        print(f"      Keys: pose {pose_key_prev}->{pose_key_new}, vel {vel_key_prev}->{vel_key_new}, bias {bias_key_prev}->{bias_key_new}")

        # CRITICAL DEBUG: Check gravity-cancelled residuals at rest
        if r == 0:  # Only debug robot 0 to avoid spam
            try:
                # --- gravity-aware sanity check (should be ~0 at rest) ---
                dt_win = float(self.preint[r].deltaTij())
                dP = np.asarray(self.preint[r].deltaPij()).reshape(3)
                dV = np.asarray(self.preint[r].deltaVij()).reshape(3)

                # Orientation at the start of the window
                try:
                    R_i = self.isam.calculateEstimatePose3(pose_key_prev).rotation()
                except Exception:
                    R_i = gtsam.Rot3()  # fallback to identity

                R = R_i.matrix()
                g_nav = np.array([0.0, 0.0, -9.81], dtype=float)

                res_v = R @ dV + g_nav * dt_win
                res_p = R @ dP + 0.5 * g_nav * (dt_win**2)

                print(f"      gravity-cancelled residuals: "
                      f"|res_v|={np.linalg.norm(res_v):.3f} m/s, "
                      f"|res_p|={np.linalg.norm(res_p):.3f} m")
                # Heuristics: at rest expect |res_v| ≲ 0.1 m/s, |res_p| ≲ 0.05 m (tune for your noise)
            except Exception as e:
                print(f"      Could not get preintegration residuals: {e}")

        # add new pose & velocity nodes initialised from previous estimates
        try:
            pose_prev = self.isam.calculateEstimatePose3(pose_key_prev)
            vel_prev  = self.isam.calculateEstimateVector(vel_key_prev)
            print(f"      Previous estimates retrieved successfully")
        except Exception as e:
            print(f"      Error retrieving previous estimates: {e}")
            pose_prev = gtsam.Pose3()  # fallback to identity
            vel_prev = np.zeros(3)
            print(f"      Using fallback values")

        self.values.insert(pose_key_new, pose_prev)     # good enough init
        self.values.insert(vel_key_new, vel_prev )
        self.values.insert(bias_key_new, gtsam.imuBias.ConstantBias())
        
        # Add ultra-weak priors for numerical stability (regularization)
        weak_v = gtsam.noiseModel.Isotropic.Sigma(3, 50.0)   # VERY weak velocity prior
        weak_b = gtsam.noiseModel.Isotropic.Sigma(6, 50.0)   # VERY weak bias prior
        self.graph.add(gtsam.PriorFactorVector(vel_key_new, np.zeros(3), weak_v))
        self.graph.add(gtsam.PriorFactorConstantBias(bias_key_new, gtsam.imuBias.ConstantBias(), weak_b))
        
        print(f"      Added new variables to values with weak priors")

        # factor graph edges - use CombinedImuFactor instead of separate ImuFactor + bias between
        print(f"      Creating CombinedImuFactor...")
        try:
            imu_factor = CombinedImuFactor(
                pose_key_prev, vel_key_prev,
                pose_key_new,  vel_key_new,
                bias_key_prev, bias_key_new,
                self.preint[r]
            )
            self.graph.add(imu_factor)
            print(f"      CombinedImuFactor added successfully")
        except Exception as e:
            print(f"      Error creating CombinedImuFactor: {e}")
            raise

        # No separate bias factor needed - it's built into CombinedImuFactor

        # reset pre‑integrator for next window
        # self.preint[r].resetIntegrationAndSetBias(imuBias.ConstantBias())
        self.last_key[r] = k_new
        
        # Record the time when this new key was created for time-based UWB attachment
        t_new = float(self.prev_imu_t[r]) if self.prev_imu_t[r] is not None else 0.0
        self.key_times[r].append(t_new)
        self.key_ids[r].append(k_new)
        
        print(f"      IMU factor creation complete for robot {r}, new last_key: {k_new}")
        
        # Flush graph to ISAM2 after completing IMU factor
        self._flush_graph()
        
        # After flushing we can read the latest bias estimate safely
        try:
            est_vals = self.isam.calculateEstimate()
            b_est = est_vals.atConstantBias(bias_key_new)
        except Exception:
            b_est = imuBias.ConstantBias()

        self.preint[r].resetIntegrationAndSetBias(b_est)

    # add range factor -----------------------------------------------------
    def _handle_uwb(self, pkt):
        sensor_kind = pkt["sensor"]   # "uwb" or "uwb_range_only"
        raw_range   = pkt["range"]
        f_id, t_id  = int(pkt["from_id"]), int(pkt["to_id"])
        
        print(f"    Processing UWB packet: robot={pkt['robot']}, from={f_id}, to={t_id}, range={raw_range:.3f}, type={sensor_kind}")

        # Simple physical gate
        if not (self.uwb_min <= raw_range <= self.uwb_max):
            print(f"      Range {raw_range:.3f}m outside bounds [{self.uwb_min}, {self.uwb_max}], dropping")
            return

        t_meas = float(pkt["timestamp"])

        # -------- A) anchor packet ------------------------------------
        if t_id <= 5:                                   # anchor IDs 0‑5
            print(f"      Anchor packet detected (to_id={t_id})")
            anchor_pos = _get_anchor_pos(self.mv.anchors, t_id)
            if anchor_pos is None:          # no calibration → skip packet
                print(f"      No anchor position found for anchor {t_id}, skipping")
                return
            print(f"      Anchor {t_id} position: {anchor_pos}")
            
            # Use from_id to determine robot for anchor ranging
            r_from = _tag_to_robot(f_id)
            if r_from is None:
                print(f"      Unknown from tag {f_id} -> skipping")
                return
            
            key_from = self._key_for_time(r_from, t_meas)
            # Get pose index for debugging
            try:
                idx = self.bisect_right(self.key_times[r_from], t_meas) - 1
                pose_idx = self.key_ids[r_from][idx] if idx >= 0 else 'unknown'
            except (IndexError, KeyError):
                pose_idx = 'unknown'
            print(f"      From key: {key_from} (robot {r_from}, pose_idx {pose_idx})")
            
            if sensor_kind == "uwb":
                cir = pkt["cir"]
                print(f"      Creating range factor (anchor, with CIR)...")
                try:
                    self.rng_factory.add_factor(
                        self.graph,
                        key_i=key_from,
                        to_id=t_id,          # anchor id
                        cir_blob=cir,
                        raw_range=raw_range
                    )
                    print(f"      Anchor range factor added successfully")
                except Exception as e:
                    print(f"      Error creating anchor range factor: {e}")
                    raise
            else:
                # range-only fallback
                print(f"      Creating range factor (anchor, range-only)...")
                try:
                    self.rng_factory.add_range_only_factor(
                        self.graph, 
                        key_i=key_from, 
                        to_id=t_id, 
                        raw_range=raw_range, 
                        sigma_m=self.uwb_sigma
                    )
                    print(f"      Anchor range-only factor added successfully")
                except Exception as e:
                    print(f"      Error creating anchor range-only factor: {e}")
                    raise
            return

        # -------- B) robot‑to‑robot -----------------------------------
        print(f"      Robot-to-robot packet detected")
        r_from = _tag_to_robot(f_id)
        r_to   = _tag_to_robot(t_id)
        if r_from is None or r_to is None:
            print(f"      Unknown tag mapping (from_id={f_id},to_id={t_id}) -> skipping")
            return

        key_from = self._key_for_time(r_from, t_meas)
        key_to   = self._key_for_time(r_to, t_meas)
        
        # Get pose indices for debugging
        try:
            idx_from = self.bisect_right(self.key_times[r_from], t_meas) - 1
            pose_idx_from = self.key_ids[r_from][idx_from] if idx_from >= 0 else 'unknown'
        except (IndexError, KeyError):
            pose_idx_from = 'unknown'
        try:
            idx_to = self.bisect_right(self.key_times[r_to], t_meas) - 1
            pose_idx_to = self.key_ids[r_to][idx_to] if idx_to >= 0 else 'unknown'
        except (IndexError, KeyError):
            pose_idx_to = 'unknown'
            
        print(f"      From robot: {r_from}, key: {key_from} (pose_idx {pose_idx_from})")
        print(f"      To robot: {r_to}, key: {key_to} (pose_idx {pose_idx_to})")

        # **Skip self‑range** (two tags on the same body frame)
        if key_to == key_from:
            print(f"      Self-range detected (same keys), skipping")
            return            # nothing useful to add to the graph

        # De-duplicate mirrored robot-to-robot factors
        pair = tuple(sorted([int(key_from), int(key_to)]))
        last_t = self.rr_last_seen.get(pair)
        if last_t is not None and abs(t_meas - last_t) < self.rr_dedup_dt:
            print(f"      Skipping near-duplicate RR factor (dt={abs(t_meas - last_t):.3f}s < {self.rr_dedup_dt}s)")
            return
        self.rr_last_seen[pair] = t_meas

        if sensor_kind == "uwb":
            cir = pkt["cir"]
            print(f"      Creating range factor (robot-to-robot, with CIR)...")
            try:
                self.rng_factory.add_factor(
                    self.graph,
                    key_i=key_from,
                    to_id=key_to,        # pose key
                    cir_blob=cir,
                    raw_range=raw_range
                )
                print(f"      Robot-to-robot range factor added successfully")
            except Exception as e:
                print(f"      Error creating robot-to-robot range factor: {e}")
                raise
        else:
            # range-only fallback
            print(f"      Creating range factor (robot-to-robot, range-only)...")
            try:
                self.rng_factory.add_range_only_factor(
                    self.graph, 
                    key_i=key_from, 
                    to_id=key_to, 
                    raw_range=raw_range, 
                    sigma_m=self.uwb_sigma
                )
                print(f"      Robot-to-robot range-only factor added successfully")
            except Exception as e:
                print(f"      Error creating robot-to-robot range-only factor: {e}")
                raise

# -------------------------------------------------------------------------
def diagnose_first_seconds(mv, seconds=10.0, robot_idx=0,
                           gyro_thr=0.02, acc_norm_thr=0.15, acc_z_thr=0.15):
    """
    Report how 'static' the first window is, and flag sustained upward proper accel.
    Uses the same accel convention as preintegration (z ≈ +g at rest).
    """
    name = list(mv.data.keys())[robot_idx]
    imu = mv.data[name]["imu_px4"].sort_values("timestamp").copy()
    if imu.empty:
        print(f"[robot {robot_idx}] no IMU data")
        return {}

    t0 = float(imu["timestamp"].iloc[0])
    imu10 = imu[imu["timestamp"] <= t0 + seconds].copy()
    if imu10.empty:
        print(f"[robot {robot_idx}] no IMU data in first {seconds}s")
        return {}

    a_raw = imu10[["linear_acceleration.x","linear_acceleration.y","linear_acceleration.z"]].to_numpy()
    w_raw = imu10[["angular_velocity.x","angular_velocity.y","angular_velocity.z"]].to_numpy()

    # FRD -> FLU, then flip sign so z ≈ +g at rest (matching your preintegration)
    a_flu   = np.stack([a_raw[:,0], -a_raw[:,1], -a_raw[:,2]], axis=1)
    a_final = -a_flu
    g = 9.81

    gyro_norm = np.linalg.norm(w_raw, axis=1)
    acc_norm  = np.linalg.norm(a_final, axis=1)
    acc_dev   = np.abs(acc_norm - g)
    az_dev    = a_final[:,2] - g  # >0 means net upward proper acceleration

    static_mask = (gyro_norm < gyro_thr) & (acc_dev < acc_norm_thr) & (np.abs(az_dev) < acc_z_thr)
    pct_static  = float(static_mask.mean()*100.0)

    # crude takeoff detection: moving average of az_dev > 0.5 m/s^2
    ts = imu10["timestamp"].to_numpy()
    dt = np.median(np.diff(ts)) if len(ts) > 1 else 0.01
    win = max(1, int(round(0.25 / max(dt, 1e-3))))  # ~0.25 s window
    kern = np.ones(win)/win
    az_ma = np.convolve(az_dev, kern, mode="same")
    idx = np.where(az_ma > 0.5)[0]
    t_takeoff = float(ts[idx[0]] - t0) if idx.size else None

    print(f"[robot {robot_idx}] first {seconds:.1f}s: static ≈ {pct_static:4.1f}%"
          f" | (az-g) min/mean/max = {az_dev.min():+.3f}/{az_dev.mean():+.3f}/{az_dev.max():+.3f} m/s^2")
    if t_takeoff is None:
        print(f"[robot {robot_idx}] no sustained upward accel (>+0.5 m/s^2) detected.")
    else:
        print(f"[robot {robot_idx}] likely takeoff at ~{t_takeoff:.2f} s.")

    return {"pct_static": pct_static, "takeoff_time_s": t_takeoff,
            "az_minus_g_min": float(az_dev.min()),
            "az_minus_g_mean": float(az_dev.mean()),
            "az_minus_g_max": float(az_dev.max())}

def range_trend_first_seconds(mv, seconds=10.0, robot_idx=0, min_samples=10):
    """
    Print Δ(gt_range) per anchor in the first 'seconds' seconds.
    Consistent non-zero deltas across multiple anchors imply motion.
    """
    name = list(mv.data.keys())[robot_idx]
    rng = mv.data[name]["uwb_range"].sort_values("timestamp").copy()
    if rng.empty:
        print(f"[robot {robot_idx}] no UWB ranges.")
        return

    t0 = float(rng["timestamp"].iloc[0])
    r10 = rng[(rng["timestamp"] >= t0) & (rng["timestamp"] <= t0 + seconds)].copy()
    if r10.empty:
        print(f"[robot {robot_idx}] no UWB ranges in first {seconds}s.")
        return

    for aid in sorted(r10["to_id"].unique()):
        # only anchors 0..5
        if int(aid) > 5:
            continue
        rr = r10[r10["to_id"] == aid]["gt_range"].dropna()
        if len(rr) < min_samples:
            continue
        delta = float(rr.iloc[-1] - rr.iloc[0])
        print(f"[robot {robot_idx}] anchor {int(aid)}: Δgt_range ≈ {delta:+.2f} m over {seconds}s (n={len(rr)})")

def rr_trend_first_seconds(mv, seconds=10.0, robot_idx=0, min_samples=10):
    """
    Δ(gt_range) for robot-to-robot links in the first 'seconds'.
    Non-zero consistent deltas across multiple links => motion.
    """
    name = list(mv.data.keys())[robot_idx]
    rng = mv.data[name]["uwb_range"].sort_values("timestamp").copy()
    if rng.empty:
        print(f"[robot {robot_idx}] no UWB ranges."); return

    t0 = float(rng["timestamp"].iloc[0])
    r10 = rng[(rng["timestamp"] >= t0) & (rng["timestamp"] <= t0 + seconds)].copy()
    if r10.empty:
        print(f"[robot {robot_idx}] no UWB ranges in first {seconds}s."); return

    # robot tags are 10/20/30; keep only robot-to-robot
    r10 = r10[(r10["to_id"] >= 10)]
    if r10.empty:
        print(f"[robot {robot_idx}] no robot-to-robot links in first {seconds}s."); return

    for peer in sorted(r10["to_id"].unique()):
        rr = r10[r10["to_id"] == peer]["gt_range"].dropna()
        if len(rr) < min_samples:
            continue
        delta = float(rr.iloc[-1] - rr.iloc[0])
        print(f"[robot {robot_idx}] to tag {int(peer)}: Δgt_range ≈ {delta:+.02f} m over {seconds}s (n={len(rr)})")

# -------------------------------------------------------------------------
class MiluvFGCoordinator:
    """Simplified coordinator that uses MILUV adapter events with existing factor graph logic."""
    
    def __init__(self, adapter, save_dir=None):
        self.adapter = adapter
        self.save_dir = save_dir
        self.static = adapter.get_static()
        
        # Initialize the same factor graph components as FGCoordinator
        print(f"=== INITIALIZING MiluvFGCoordinator ===")
        
        # IMU preintegration setup (same as FGCoordinator)
        g = 9.81
        params = gtsam.PreintegrationCombinedParams.MakeSharedU(g)
        if hasattr(params, "setGravityVector"):
            params.setGravityVector(np.array([0.0, 0.0, -g], dtype=float))
        params.setAccelerometerCovariance(np.eye(3) * (0.03**2))
        params.setGyroscopeCovariance(np.eye(3) * (0.002**2))
        params.setIntegrationCovariance(np.eye(3) * ((1e-3)**2))
        if hasattr(params, "setBiasAccCovariance"):
            params.setBiasAccCovariance(np.eye(3) * 1e-4)
        if hasattr(params, "setBiasOmegaCovariance"):
            params.setBiasOmegaCovariance(np.eye(3) * 1e-6)
        if hasattr(params, "setBiasAccOmegaIntCovariance"):
            params.setBiasAccOmegaIntCovariance(np.eye(6) * 1e-5)
        
        self.preint = {}
        self.prev_imu_t = {}
        robot_count = self.static['robot_count']
        for r in range(robot_count):
            self.preint[r] = gtsam.PreintegratedCombinedMeasurements(params, imuBias.ConstantBias())
            self.prev_imu_t[r] = None
        
        # Key tracking
        import bisect
        self.bisect_right = bisect.bisect_right
        self.key_times = {r: [] for r in range(robot_count)}
        self.key_ids = {r: [] for r in range(robot_count)}
        
        # Initialize with t=0 keys
        for r in range(robot_count):
            self.key_times[r].append(0.0)
            self.key_ids[r].append(0)
        
        # Range factory with UWB ML
        exp_name = self.adapter.exp_name
        self.rng_factory = RangeFactoryStd(UwbML(exp_name))
        
        # RR dedup
        self.rr_last_seen = {}
        self.rr_dedup_dt = self.adapter.rr_dedup_dt
        
        # Factor graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        
        params = gtsam.ISAM2Params()
        if hasattr(gtsam.ISAM2Params, 'QR'):
            params.setFactorization(gtsam.ISAM2Params.QR)
        if hasattr(params, 'setRelinearizeThreshold'):
            params.setRelinearizeThreshold(0.01)
        elif hasattr(params, 'relinearizeThreshold'):
            params.relinearizeThreshold = 0.01
        if hasattr(params, 'setRelinearizeSkip'):
            params.setRelinearizeSkip(1)
        elif hasattr(params, 'relinearizeSkip'):
            params.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(params)
        
        # Load anchors
        if self.static['anchors']:
            self.rng_factory.load_anchors(self.graph, self.values, self.static['anchors'])
        
        # Add priors
        self._add_priors()
        
        # Initialize last_key tracking
        self.last_key = {r: 0 for r in range(robot_count)}
        
        # Bootstrap
        self._flush_graph()
        print("=== MiluvFGCoordinator INITIALIZED ===")
    
    def _add_priors(self):
        """Add pose, velocity, and bias priors."""
        robot_count = self.static['robot_count']
        
        for ridx in range(robot_count):
            k = POSE_KEY(ridx, 0)
            R0 = gtsam.Rot3()  # Identity rotation for simplicity
            P0 = gtsam.Pose3(R0, gtsam.Point3(0, 0, 0))
            
            if ridx == 0:
                prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]*6))
            else:
                sig = np.array([0.05, 0.05, np.pi/3, 1.0, 1.0, 1.0], float)
                prior_noise = gtsam.noiseModel.Diagonal.Sigmas(sig)
            
            self.values.insert(k, P0)
            self.graph.add(gtsam.PriorFactorPose3(k, P0, prior_noise))
            
            # Velocity and bias priors
            vel_key = VEL_KEY(ridx, 0)
            bias_key = BIAS_KEY(ridx, 0)
            
            self.values.insert(vel_key, np.zeros(3))
            self.values.insert(bias_key, gtsam.imuBias.ConstantBias())
            
            v_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
            b_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            
            self.graph.add(gtsam.PriorFactorVector(vel_key, np.zeros(3), v_noise))
            self.graph.add(gtsam.PriorFactorConstantBias(bias_key, gtsam.imuBias.ConstantBias(), b_noise))
    
    def _flush_graph(self):
        """Push accumulated factors to ISAM2."""
        if self.graph.size() == 0:
            return
        self.isam.update(self.graph, self.values)
        self.graph = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
    
    def _key_for_time(self, robot: int, t: float) -> int:
        """Get the pose key for robot at time t."""
        ts = self.key_times[robot]
        ks = self.key_ids[robot]
        if not ts:
            return POSE_KEY(robot, self.last_key[robot])
        idx = self.bisect_right(ts, t) - 1
        if idx < 0:
            idx = 0
        return POSE_KEY(robot, ks[idx])
    
    def run_miluv_events(self, event_stream, max_sec=30):
        """Process MILUV events through the factor graph."""
        print(f"\n=== STARTING MILUV FACTOR GRAPH PROCESSING (max {max_sec} seconds) ===")
        
        t0 = time.time()
        packet_count = 0
        imu_count = 0
        uwb_count = 0
        height_count = 0
        
        for typ, pkt in event_stream:
            packet_count += 1
            elapsed = time.time() - t0
            
            if packet_count % 100 == 0:
                print(f"  Processed {packet_count} packets in {elapsed:.2f}s (IMU: {imu_count}, UWB: {uwb_count}, HEIGHT: {height_count})")
                print(f"    Current graph size: {self.graph.size()} factors")
            
            if typ == "IMU":
                imu_count += 1
                self._handle_miluv_imu(pkt)
            elif typ == "UWB":
                uwb_count += 1
                self._handle_miluv_uwb(pkt)
            elif typ == "HEIGHT":
                height_count += 1
                # TODO: implement height handling if needed
                pass
            
            if time.time() - t0 > max_sec:
                print(f"  Time limit reached after {packet_count} packets")
                break
        
        # Final flush
        self._flush_graph()
        
        print(f"=== MILUV FACTOR GRAPH PROCESSING COMPLETE ===")
        print(f"  Total packets processed: {packet_count} (IMU: {imu_count}, UWB: {uwb_count}, HEIGHT: {height_count})")
        print(f"  Final ISAM2 estimates: {self.isam.calculateEstimate().size()} variables")
    
    def _handle_miluv_imu(self, pkt):
        """Handle IMU events from MILUV adapter."""
        # Convert ImuEvent to the format expected by existing logic
        r = pkt.robot
        
        if self.prev_imu_t[r] is not None:
            dt = pkt.t - self.prev_imu_t[r]
            
            # Convert PX4 FRD → FLU (match non-MILUV path)
            accel_body = frd_to_flu(pkt.accel)
            gyro_body = frd_to_flu(pkt.omega)
            
            # Specific force expected by our preintegration pipeline:
            # at rest (body FLU), accel ≈ [0, 0, +g]
            accel_meas = -accel_body
            gyro_meas = gyro_body
            
            self.preint[r].integrateMeasurement(accel_meas, gyro_meas, dt)
        
        self.prev_imu_t[r] = pkt.t
        
        # Create IMU factor every ~0.75s
        delta_t = self.preint[r].deltaTij()
        if delta_t < 0.75:
            return
        
        k_prev = self.last_key[r]
        k_new = k_prev + 1
        
        pose_key_prev = POSE_KEY(r, k_prev)
        vel_key_prev = VEL_KEY(r, k_prev)
        bias_key_prev = BIAS_KEY(r, k_prev)
        pose_key_new = POSE_KEY(r, k_new)
        vel_key_new = VEL_KEY(r, k_new)
        bias_key_new = BIAS_KEY(r, k_new)
        
        # Initialize new variables
        try:
            pose_prev = self.isam.calculateEstimatePose3(pose_key_prev)
            vel_prev = self.isam.calculateEstimateVector(vel_key_prev)
        except Exception:
            pose_prev = gtsam.Pose3()
            vel_prev = np.zeros(3)
        
        self.values.insert(pose_key_new, pose_prev)
        self.values.insert(vel_key_new, vel_prev)
        self.values.insert(bias_key_new, gtsam.imuBias.ConstantBias())
        
        # Weak priors for regularization
        weak_v = gtsam.noiseModel.Isotropic.Sigma(3, 50.0)
        weak_b = gtsam.noiseModel.Isotropic.Sigma(6, 50.0)
        self.graph.add(gtsam.PriorFactorVector(vel_key_new, np.zeros(3), weak_v))
        self.graph.add(gtsam.PriorFactorConstantBias(bias_key_new, gtsam.imuBias.ConstantBias(), weak_b))
        
        # IMU factor
        imu_factor = CombinedImuFactor(
            pose_key_prev, vel_key_prev,
            pose_key_new, vel_key_new,
            bias_key_prev, bias_key_new,
            self.preint[r]
        )
        self.graph.add(imu_factor)
        
        # Update tracking
        self.last_key[r] = k_new
        self.key_times[r].append(float(pkt.t))
        self.key_ids[r].append(k_new)
        
        # Flush and reset
        self._flush_graph()
        
        try:
            est_vals = self.isam.calculateEstimate()
            b_est = est_vals.atConstantBias(bias_key_new)
        except Exception:
            b_est = imuBias.ConstantBias()
        
        self.preint[r].resetIntegrationAndSetBias(b_est)
    
    def _handle_miluv_uwb(self, pkt):
        """Handle UWB events from MILUV adapter."""
        # Convert UwbEvent to existing format
        raw_range = pkt.rng
        f_id, t_id = pkt.from_id, pkt.to_id
        t_meas = pkt.t
        
        # Anchor ranging
        if t_id <= 5:
            anchor_pos = self.static['anchors'].get(t_id)
            if anchor_pos is None:
                return
            
            r_from = _tag_to_robot(f_id)
            if r_from is None:
                return
            
            key_from = self._key_for_time(r_from, t_meas)
            
            # Use range-only factor (since CIR processing is complex)
            self.rng_factory.add_range_only_factor(
                self.graph,
                key_i=key_from,
                to_id=t_id,
                raw_range=raw_range,
                sigma_m=pkt.std
            )
            return
        
        # Robot-to-robot ranging
        r_from = _tag_to_robot(f_id)
        r_to = _tag_to_robot(t_id)
        if r_from is None or r_to is None:
            return
        
        key_from = self._key_for_time(r_from, t_meas)
        key_to = self._key_for_time(r_to, t_meas)
        
        if key_to == key_from:
            return  # Skip self-range
        
        # Dedup
        pair = tuple(sorted([int(key_from), int(key_to)]))
        last_t = self.rr_last_seen.get(pair)
        if last_t is not None and abs(t_meas - last_t) < self.rr_dedup_dt:
            return
        self.rr_last_seen[pair] = t_meas
        
        # Add range factor
        self.rng_factory.add_range_only_factor(
            self.graph,
            key_i=key_from,
            to_id=key_to,
            raw_range=raw_range,
            sigma_m=pkt.std
        )

# -------------------------------------------------------------------------
def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog = "fg_runner.py",
        description = "Realtime factor‑graph demo based on ISAM2"
    )
    p.add_argument("dataset", help="dataset id (e.g., cir_3_random3_0 or miluv:<exp_name>)")
    p.add_argument("--max-sec", type=float, default=20,
                   help="wall‑clock limit (default: 20 seconds)")
    p.add_argument("--save-dir", type=str, default=None,          ### NEW
                   help="dump every incremental graph/values here")
    p.add_argument("--csv-out", type=str, default=None,           ### NEW
                   help="CSV file to store final ISAM2 poses")
    p.add_argument("--cir-dt", type=float, default=0.15,
                   help="CIR/Range fuse tolerance in seconds (default 0.15)")
    p.add_argument("--uwb-sigma", type=float, default=0.25,
                   help="Std dev for range-only factors in meters (default 0.25)")
    p.add_argument("--uwb-min", type=float, default=0.05,
                   help="Min plausible UWB range in meters (default: 0.05)")
    p.add_argument("--uwb-max", type=float, default=25.0,
                   help="Max plausible UWB range in meters (default: 25.0)")
    p.add_argument("--rr-dedup-dt", type=float, default=0.02,
                   help="Time window (s) to deduplicate mirrored RR ranges")
    
    # MILUV-only knobs (all optional):
    p.add_argument("--miluv-root", default="./data", help="root folder containing MILUV data/<exp>")
    p.add_argument("--imu-source", choices=["px4","cam","both"], default="px4")
    p.add_argument("--miluv-remove-imu-bias", action="store_true", default=True)
    p.add_argument("--no-miluv-remove-imu-bias", dest="miluv_remove_imu_bias", action="store_false")
    p.add_argument("--use-height", action="store_true", default=False)
    return p

if __name__ == "__main__":
    import os
    
    args = build_arg_parser().parse_args()
    os.environ["GTSAM_USE_QUATERNIONS"]="1"
    
    if args.save_dir:
        args.save_dir = ensure_dir(args.save_dir)
    
    is_miluv = isinstance(args.dataset, str) and args.dataset.startswith("miluv:")
    
    if is_miluv:
        # === MILUV path ===
        assert MiluvAdapter is not None, "sota.miluv.loader import failed"
        exp_name = args.dataset.split(":",1)[1]
        adapter = MiluvAdapter(
            exp_name=exp_name,
            exp_dir=args.miluv_root,
            imu_source=args.imu_source,
            include_height=args.use_height,
            remove_imu_bias=args.miluv_remove_imu_bias,
            uwb_sigma_default=args.uwb_sigma,
            uwb_min=args.uwb_min,
            uwb_max=args.uwb_max,
            rr_dedup_dt=args.rr_dedup_dt,
        )
        static = adapter.get_static()
        event_stream = adapter.iter_events(t0=0.0, t1=args.max_sec)
        
        print(f"=== MILUV ADAPTER INITIALIZED ===")
        print(f"Experiment: {exp_name}")
        print(f"Robot count: {static['robot_count']}")
        print(f"Anchors: {list(static['anchors'].keys()) if static['anchors'] else 'None'}")
        print(f"Tag offsets: {list(static['tag_offsets'].keys()) if static['tag_offsets'] else 'None'}")
        
        # Create a MILUV-compatible coordinator using the adapter data
        coord = MiluvFGCoordinator(adapter, save_dir=args.save_dir)
        
        # Run the factor graph with MILUV events
        t0 = time.time()
        coord.run_miluv_events(event_stream, max_sec=args.max_sec)
        dt = time.time() - t0
        print(f"Finished MILUV processing in {dt:.2f}s")
        
        if args.csv_out:
            # new: include timestamps
            from sota.factor_graph.utils_io import save_estimates_with_timestamps
            save_estimates_with_timestamps(
                coord.isam,
                args.csv_out,
                key_times=coord.key_times,
                key_ids=coord.key_ids,
                robot_syms=[chr(ord('P')+r) for r in range(static['robot_count'])]
            )
            print(f"ISAM2 poses saved to {args.csv_out}")
        
    else:
        # === Original path ===
        coord = FGCoordinator(args.dataset, save_dir=args.save_dir, cir_dt=args.cir_dt, uwb_sigma=args.uwb_sigma,
                              uwb_min=args.uwb_min, uwb_max=args.uwb_max, rr_dedup_dt=args.rr_dedup_dt)
        
        # Run diagnostics before factor graph processing
        print("\n=== DIAGNOSTICS: First 10 seconds analysis ===")
        for ridx in (0,1,2):
            print(f"\n--- Robot {ridx} IMU Analysis ---")
            diagnose_first_seconds(coord.mv, seconds=10.0, robot_idx=ridx)
            print(f"\n--- Robot {ridx} Anchor Range Trend Analysis ---")
            range_trend_first_seconds(coord.mv, seconds=10.0, robot_idx=ridx)
            print(f"\n--- Robot {ridx} Robot-to-Robot Range Trend Analysis ---")
            rr_trend_first_seconds(coord.mv, seconds=10.0, robot_idx=ridx)
        print("\n=== Starting Factor Graph Processing ===")
        
        t0 = time.time()
        coord.run(max_sec=args.max_sec)
        dt = time.time() - t0
        print(f"Finished in {dt:.2f}s")
        
        if args.csv_out:                         ### NEW
            save_estimates(coord.isam, args.csv_out)
            print(f"ISAM2 poses saved to {args.csv_out}")