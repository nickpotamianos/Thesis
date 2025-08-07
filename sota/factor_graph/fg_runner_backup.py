# sota/factor_graph/fg_runner.py
import time, yaml, gtsam, numpy as np
import pandas as pd
from pathlib import Path
from miluv.data import DataLoader
from gtsam import (
    PreintegratedImuMeasurements, ImuFactor, CombinedImuFactor,
    imuBias, noiseModel
)
from sota.uwb.infer_uwb          import UwbML
from sota.factor_graph.range_factory_std import RangeFactoryStd
from sota.factor_graph.utils_io  import ensure_dir, save_estimates, save_graph_values   ### NEW

def frd_to_flu(v):
    """PX4 FRD -> FLU (keep x, flip y and z). Works for accel and gyro."""
    return np.array([v[0], -v[1], -v[2]], dtype=float)

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
def fuse_range_and_cir(robot_table, max_dt=0.05):
    """
    Return a DataFrame that contains   timestamp, from_id, to_id,
    range, gt_range, cir    – i.e. every range sample augmented with
    the closest CIR sample that has the *same link direction* and is
    within ±max_dt seconds (default 50 ms).
    """
    rng = robot_table["uwb_range"][["timestamp",
                                    "from_id", "to_id",
                                    "range", "gt_range"]].copy()
    cir = robot_table["uwb_cir"  ][["timestamp",
                                    "from_id", "to_id",
                                    "cir"]].copy()

    rng = rng.sort_values("timestamp")
    cir = cir.sort_values("timestamp")

    fused = pd.merge_asof(
        rng, cir,
        on       = "timestamp",
        by       = ["from_id", "to_id"],
        tolerance= max_dt,
        direction= "nearest"
    )

    # drop rows where no CIR was found
    fused = fused.dropna(subset=["cir"])

    return fused

POSE_KEY   = lambda r, k: gtsam.symbol(chr(ord('P') + r), k)   # P,Q,R
VEL_KEY    = lambda r, k: gtsam.symbol(chr(ord('V') + r), k)   # V,W,X
BIAS_KEY   = lambda r, k: gtsam.symbol(chr(ord('B') + r), k)   # B,C,D

class FGCoordinator:
    def __init__(self, exp, save_dir=None):                    ### NEW
        print(f"=== INITIALIZING FGCoordinator for experiment: {exp} ===")
        self.exp  = exp
        self.save_dir = save_dir                               ### NEW
        self.packet_counter = 0                                ### NEW
        self.mv   = DataLoader(exp, exp_dir="./data/three_robots", cir=True, height=False)   # height unused for now   
        print(f"DataLoader created successfully")
        print(f"Available robots: {list(self.mv.data.keys())}")
        
        # flatten all anchor vectors in place
        if self.mv.anchors:
            print(f"Raw anchors before flattening: {self.mv.anchors}")
            for k, v in self.mv.anchors.items():
                old_shape = np.asarray(v).shape
                self.mv.anchors[k] = np.asarray(v, dtype=float).reshape(-1)
                print(f"  Anchor {k}: {old_shape} -> {self.mv.anchors[k].shape}, value: {self.mv.anchors[k]}")
        else:
            print("No anchors found in dataset")
            
        # Complete initialization
        self.__init_rest__()

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
        a = frd_to_flu(np.array([ax, ay, az], dtype=float))  # now FLU

        # Expected at rest (FLU, z up): a ≈ [0, 0, -g]
        # Roll/pitch that align body z with world z:
        roll  = np.arctan2(a[1], a[2])
        pitch = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
        yaw   = 0.0
        return gtsam.Rot3.Ypr(yaw, pitch, roll)  # GTSAM uses Y,P,R order here
        
    def __init_rest__(self):
        # IMU pre‑integration parameters (calibrated once)
        print("=== SETTING UP IMU PRE-INTEGRATION ===")
        imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
        imu_params.setGyroscopeCovariance(
            np.eye(3)* (0.015**2))      # rad²/s²
        imu_params.setAccelerometerCovariance(
            np.eye(3)* (0.019**2))      # (m/s²)²
        imu_params.setIntegrationCovariance(
            np.eye(3)* (0.0001**2))
        print(f"IMU params created with gravity: {9.81}")
        print(f"Gyro covariance: {0.015**2}")
        print(f"Accel covariance: {0.019**2}")

        self.preint      = {}
        self.prev_imu_t  = {}
        for r in range(3):
            self.preint[r]     = PreintegratedImuMeasurements(
                                    imu_params,
                                    imuBias.ConstantBias())
            self.prev_imu_t[r] = None
            print(f"  Robot {r}: IMU preintegrator initialized")
        
        print("=== SETTING UP UWB ML FRONTEND ===")
        self.rng_factory = RangeFactoryStd(UwbML(self.exp))          # pre‑trained CNNs
        print(f"RangeFactoryStd created for experiment: {self.exp}")
        
        # Debug: check what anchors are loaded
        print("=== ANCHOR SUMMARY ===")
        print("Anchors loaded:", sorted(self.mv.anchors.keys()) if self.mv.anchors else "None")
        if self.mv.anchors:
            for k, v in self.mv.anchors.items():
                print(f"  Anchor {k}: {v} (type: {type(v)}, shape: {v.shape})")
        
        print("=== SETTING UP FACTOR GRAPH ===")
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam   = gtsam.ISAM2()

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
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3]*6))                              # pose6d
        print(f"Prior noise model: {prior_noise}")
        k0 = 0
        for ridx, robot in enumerate(self.mv.data.keys()):
            k = POSE_KEY(ridx, k0)
            R0 = self._estimate_initial_orientation(ridx)    # <- NEW
            P0 = gtsam.Pose3(R0, gtsam.Point3(0, 0, 0))
            print(f"  Robot {ridx} ({robot}): pose key = {k}, init tilt from accel")
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
        print("=== INITIALIZATION COMPLETE ===\n")
            for k, v in self.mv.anchors.items():
                print(f"  Anchor {k}: {v} (type: {type(v)}, shape: {v.shape})")
        
        print("=== SETTING UP FACTOR GRAPH ===")
        self.graph  = gtsam.NonlinearFactorGraph()
        self.values = gtsam.Values()
        self.isam   = gtsam.ISAM2()

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
        prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-3]*6))                              # pose6d
        print(f"Prior noise model: {prior_noise}")
        k0 = 0
        for ridx, robot in enumerate(self.mv.data.keys()):
            k = POSE_KEY(ridx, k0)
            R0 = self._estimate_initial_orientation(ridx)    # <- NEW
            P0 = gtsam.Pose3(R0, gtsam.Point3(0, 0, 0))
            print(f"  Robot {ridx} ({robot}): pose key = {k}, init tilt from accel")
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
        print("=== INITIALIZATION COMPLETE ===\n")

    def _flush_graph(self):
        """
        Push the factors accumulated in self.graph/self.values into ISAM2,
        then clear the buffers so the next window starts empty.
        """
        if self.graph.size() == 0:
            return                                             # nothing to do

        print(f"    Flushing graph with {self.graph.size()} factors, {self.values.size()} values to ISAM2")
        
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
            uwb = fuse_range_and_cir(rob)           # <‑‑ new helper
            uwb["sensor"] = "uwb"
            uwb["robot"]  = ridx
            print(f"      UWB fused packets: {len(uwb)}")
            total_uwb_packets += len(uwb)
            rows.append(uwb)

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
            else:
                uwb_count += 1
                if uwb_count <= 5:  # Detailed debug for first few UWB packets
                    print(f"    UWB packet {uwb_count}: robot={pkt['robot']}, from={pkt['from_id']}, to={pkt['to_id']}, range={pkt['range']:.3f}")
                try:
                    self._handle_uwb(pkt)
                except Exception as e:
                    print(f"    ERROR in _handle_uwb: {e}")
                    print(f"    Packet details: {pkt}")
                    raise
            
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
            accel_frd = np.array([pkt["linear_acceleration.x"],
                                  pkt["linear_acceleration.y"],
                                  pkt["linear_acceleration.z"]], dtype=float)
            gyro_frd  = np.array([pkt["angular_velocity.x"],
                                  pkt["angular_velocity.y"],
                                  pkt["angular_velocity.z"]], dtype=float)

            accel = frd_to_flu(accel_frd)   # <- FIX
            gyro  = frd_to_flu(gyro_frd)    # <- FIX
            
            if r == 0 and self.preint[r].deltaTij() < 0.1:  # Debug first robot, first few measurements
                print(f"      IMU integration robot {r}: dt={dt:.6f}, accel={accel}, gyro={gyro}")
            
            self.preint[r].integrateMeasurement(accel, gyro, dt)
        else:
            print(f"      First IMU measurement for robot {r} at t={pkt['timestamp']:.6f}")
            
        self.prev_imu_t[r] = pkt["timestamp"]

        # every 200 samples (≈2 s at 100 Hz) create a factor
        delta_t = self.preint[r].deltaTij()
        if delta_t < 2.0:
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
        
        print(f"      Added new variables to values")

        # factor graph edges
        print(f"      Creating ImuFactor...")
        try:
            imu_factor = ImuFactor(
                pose_key_prev, vel_key_prev,
                pose_key_new,  vel_key_new,
                bias_key_prev,
                self.preint[r])
            self.graph.add(imu_factor)
            print(f"      ImuFactor added successfully")
        except Exception as e:
            print(f"      Error creating ImuFactor: {e}")
            raise

        # bias random‑walk
        print(f"      Creating bias factor...")
        try:
            bias_covar = noiseModel.Isotropic.Sigma(6, 0.0001)
            bias_factor = gtsam.BetweenFactorConstantBias(
                bias_key_prev, bias_key_new,
                imuBias.ConstantBias(), bias_covar)
            self.graph.add(bias_factor)
            print(f"      Bias factor added successfully")
        except Exception as e:
            print(f"      Error creating bias factor: {e}")
            raise

        # reset pre‑integrator for next window
        # self.preint[r].resetIntegrationAndSetBias(imuBias.ConstantBias())
        self.last_key[r] = k_new
        print(f"      IMU factor creation complete for robot {r}, new last_key: {k_new}")
        
        # Flush graph to ISAM2 after completing IMU factor
        self._flush_graph()
        
        # After flushing we can read the latest bias estimate safely
        try:
            est_vals = self.isam.calculateEstimate()
            b_est = est_vals.atConstantBias(bias_key_new)
        except Exception:
            b_est = gtsam.imuBias.ConstantBias()

        self.preint[r].resetIntegrationAndSetBias(b_est)

    # add range factor -----------------------------------------------------
    def _handle_uwb(self, pkt):
        print(f"    Processing UWB packet: robot={pkt['robot']}, from={pkt['from_id']}, to={pkt['to_id']}, range={pkt['range']:.3f}")
        
        cir        = pkt["cir"]
        raw_range  = pkt["range"]
        f_id, t_id = int(pkt["from_id"]), int(pkt["to_id"])

        key_from = POSE_KEY(pkt["robot"], self.last_key[pkt["robot"]])
        print(f"      From key: {key_from} (robot {pkt['robot']}, pose_idx {self.last_key[pkt['robot']]})")

        # -------- A) anchor packet ------------------------------------
        if t_id <= 5:                                   # anchor IDs 0‑5
            print(f"      Anchor packet detected (to_id={t_id})")
            anchor_pos = _get_anchor_pos(self.mv.anchors, t_id)
            if anchor_pos is None:          # no calibration → skip packet
                print(f"      No anchor position found for anchor {t_id}, skipping")
                return
            print(f"      Anchor {t_id} position: {anchor_pos}")
            
            print(f"      Creating range factor (anchor)...")
            try:
                self.rng_factory.add_factor(
                    self.graph,
                    key_i=key_from,
                    to_id=t_id,          # anchor id
                    cir_blob=cir,
                    raw_range=raw_range
                )
                print(f"      Anchor range factor added successfully")
                self._flush_graph()  # Flush after adding anchor range factor
            except Exception as e:
                print(f"      Error creating anchor range factor: {e}")
                print(f"      CIR type: {type(cir)}, shape: {getattr(cir, 'shape', 'N/A')}")
                print(f"      Raw range: {raw_range}")
                print(f"      Anchor pos: {anchor_pos}")
                raise
            return

        # -------- B) robot‑to‑robot -----------------------------------
        print(f"      Robot-to-robot packet detected")
        r_to = [0, 1, 2][[10, 20, 30].index(t_id//10*10)]
        key_to = POSE_KEY(r_to, self.last_key[r_to])
        print(f"      To robot: {r_to}, key: {key_to} (pose_idx {self.last_key[r_to]})")

        # **Skip self‑range** (two tags on the same body frame)
        if key_to == key_from:
            print(f"      Self-range detected (same keys), skipping")
            return            # nothing useful to add to the graph

        print(f"      Creating range factor (robot-to-robot)...")
        try:
            self.rng_factory.add_factor(
                self.graph,
                key_i=key_from,
                to_id=key_to,        # pose key
                cir_blob=cir,
                raw_range=raw_range
            )
            print(f"      Robot-to-robot range factor added successfully")
            self._flush_graph()  # Flush after adding robot-to-robot range factor
        except Exception as e:
            print(f"      Error creating robot-to-robot range factor: {e}")
            print(f"      CIR type: {type(cir)}, shape: {getattr(cir, 'shape', 'N/A')}")
            print(f"      Raw range: {raw_range}")
            print(f"      From key: {key_from}, To key: {key_to}")
            raise

# -------------------------------------------------------------------------
def build_arg_parser():
    import argparse
    p = argparse.ArgumentParser(
        prog = "fg_runner.py",
        description = "Realtime factor‑graph demo based on ISAM2"
    )
    p.add_argument("exp", help="experiment name / directory")
    p.add_argument("--max-sec", type=float, default=20,
                   help="wall‑clock limit (default: 20 seconds)")
    p.add_argument("--save-dir", type=str, default=None,          ### NEW
                   help="dump every incremental graph/values here")
    p.add_argument("--csv-out", type=str, default=None,           ### NEW
                   help="CSV file to store final ISAM2 poses")
    return p

if __name__ == "__main__":
    import os
    
    args = build_arg_parser().parse_args()
    os.environ["GTSAM_USE_QUATERNIONS"]="1"
    
    if args.save_dir:
        args.save_dir = ensure_dir(args.save_dir)
    
    coord = FGCoordinator(args.exp, save_dir=args.save_dir)
    t0 = time.time()
    coord.run(max_sec=args.max_sec)
    dt = time.time() - t0
    print(f"Finished in {dt:.2f}s")
    
    if args.csv_out:                         ### NEW
        save_estimates(coord.isam, args.csv_out)
        print(f"ISAM2 poses saved to {args.csv_out}")
