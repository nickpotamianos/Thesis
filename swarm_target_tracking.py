# swarm_target_tracking.py
import argparse, os, json
import numpy as np
import pandas as pd

# ---- Authors' devkit imports (do not modify) ----
from miluv.data import DataLoader
import miluv.utils as utils

# The EKF models live in the examples package in upstream devkit.
try:
    import examples.ekfutils.imu_three_robots_models as model
except Exception:
    import imu_three_robots_models as model  # fallback to local path if exported

# ---- Our modules (new) ----
from swarm_ml.roles import get_roles
from swarm_ml.features import se_translation_from_matrix, build_measurement_features
from swarm_ml.measure_adapter import MeasureAdapter, AdapterConfig
from swarm_ml.target_filter import TargetIF, IFConfig
from swarm_ml.fusion import CIFuser, CIFuserConfig
from swarm_ml.evaluation_swarm import evaluate_and_save
from swarm_ml.tagmap import infer_tag_ids_by_robot, select_pair_rows, robust_range_aggregate, robust_tracker_sensor_position, robust_target_offset
from swarm_ml.smoother import rts_smooth, CVNoise
from swarm_ml.los_adapter import LOSAdapter, LOSConfig
from swarm_ml.online_tuner import OnlineTuner, OnlineAdaptConfig
from swarm_ml.snapshots import SnapshotCollector

def _concat_with_robot(data: dict, key: str) -> pd.DataFrame:
    dfs = []
    for robot in data.keys():
        if key in data[robot]:
            dfs.append(data[robot][key].assign(robot=robot))
    if not dfs:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(dfs, ignore_index=True)

def tag_world_position(T_rb: np.ndarray, tag_id: int, tag_moment_arms) -> np.ndarray:
    """
    Compute world position of a tag mounted on the tracker robot body.
    - T_rb: 4x4 or 5x5 pose (R|p)
    - tag_moment_arms: miluv.tag_moment_arms (assumed dict-like)
    Fallback: return robot position if mapping not found.
    """
    try:
        R = T_rb[:3, :3]
        p = T_rb[:3, -1]
        # Attempt flexible lookups:
        # tag_moment_arms may be dict[int] -> (x,y,z) OR nested by robot id
        if isinstance(tag_moment_arms, dict):
            if tag_id in tag_moment_arms:
                arm = np.asarray(tag_moment_arms[tag_id], dtype=float).reshape(3)
            else:
                # scan nested dicts for the tag_id
                arm = None
                for v in tag_moment_arms.values():
                    if isinstance(v, dict) and tag_id in v:
                        arm = np.asarray(v[tag_id], dtype=float).reshape(3); break
                if arm is None:
                    return p
        else:
            return p
        return p + R @ arm
    except Exception:
        return T_rb[:3, -1]

def load_biasnet(path_dir: str):
    import json, torch
    from swarm_ml.models import BiasNet
    with open(os.path.join(path_dir, "biasnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    model = BiasNet(in_dim)
    state = torch.load(os.path.join(path_dir, "biasnet.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def load_fusionnet(path_dir: str):
    import json, torch
    from swarm_ml.models import FusionNet
    with open(os.path.join(path_dir, "fusionnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    model = FusionNet(in_dim)
    state = torch.load(os.path.join(path_dir, "fusionnet.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model

def main(args):
    exp_name = args.exp
    os.makedirs(args.out, exist_ok=True)

    # ----------------- Load data (authors' devkit) -----------------
    miluv = DataLoader(
        exp_name,
        exp_dir="./data/three_robots",  # keep your local path
        cir=False,
        barometer=False,
        height=True if (args.use_height or args.use_height_tf) else False,
        imu="px4",
        cam=None,
        mag=False
    )
    data = miluv.data
    robots = list(data.keys())

    # Inter-robot UWB ranges + (optional) height
    uwb_range = _concat_with_robot(data, "uwb_range")
    height_df = _concat_with_robot(data, "height") if (args.use_height or args.use_height_tf) else pd.DataFrame(columns=["timestamp"])

    # Query timestamps (union of UWB and height times)
    query_timestamps = np.sort(np.unique(np.append(
        uwb_range["timestamp"].to_numpy(),
        height_df["timestamp"].to_numpy() if not height_df.empty else np.array([], dtype=float)
    )))

    # IMU at query timestamps (authors' API)
    imu_at_q = {
        robot: miluv.query_by_timestamps(query_timestamps, robots=robot, sensors="imu_px4")[robot]
        for robot in robots
    }
    gyro = {
        robot: imu_at_q[robot]["imu_px4"][["timestamp", "angular_velocity.x", "angular_velocity.y", "angular_velocity.z"]].reset_index(drop=True)
        for robot in robots
    }
    accel = {
        robot: imu_at_q[robot]["imu_px4"][["timestamp", "linear_acceleration.x", "linear_acceleration.y", "linear_acceleration.z"]].reset_index(drop=True)
        for robot in robots
    }

    # Ground truth (authors' tools)
    gt_se23 = {
        robot: utils.get_se23_poses(
            data[robot]["mocap_quat"](query_timestamps),
            data[robot]["mocap_pos"].derivative(nu=1)(query_timestamps),
            data[robot]["mocap_pos"](query_timestamps)
        )
        for robot in robots
    }

    # ----------------- Initialize authors' multi-robot EKF -----------------
    ekf_history = {
        robot: {
            "pose": model.common.MatrixStateHistory(state_dim=5, covariance_dim=9),
            "bias": model.common.VectorStateHistory(state_dim=6),
        }
        for robot in robots
    }

    ekf = model.EKF(
        {robot: gt_se23[robot][0] for robot in robots},  # authors' initialization from GT
        miluv.anchors,
        miluv.tag_moment_arms
    )

    # ----------------- Our swarm setup -----------------
    roles = get_roles(exp_name, robots, default_target=args.target, uwb_range_df=uwb_range)
    print(f"[SWARM] Target: {roles.target}; Trackers: {roles.trackers}")

    # Infer tag IDs per robot for proper tracker↔target filtering
    tag_map = infer_tag_ids_by_robot(uwb_range, robots, top_n=args.tags_per_robot)
    print(f"[SWARM] Inferred tag IDs per robot: {tag_map}")

    # Per-tracker target filters (consider tighter gate for robustness)
    tf_cfg = IFConfig(
        sigma_a_xy=args.sigma_a_xy,
        sigma_a_z=args.sigma_a_z,
        p0_xy=args.p0, p0_z=args.p0,
        v0_xy=args.v0, v0_z=args.v0,
        gate_N_sigma=args.gate_sigma
    )

    # Optional: robust target initialization from first window
    from swarm_ml.target_init import init_target_from_window
    from functools import partial
    from swarm_ml.tagmap import robust_range_aggregate

    p0_target = None
    if args.init_window > 0:
        def tracker_pose_fn(trk, t):
            # use current EKF pose at the *closest* query time; here we'll assume exact t
            T_trk = ekf.pose.get(trk, None)
            if T_trk is None:
                return np.zeros(3)
            # you can reuse tag_world_position to pick the dominant tag if you want
            return T_trk[:3, -1]
        agg = lambda df_pair: robust_range_aggregate(df_pair, base_var=args.uwb_var,
                                                     rho=args.pair_corr, huber_delta=args.huber_delta)
        p0_target = init_target_from_window(query_timestamps, uwb_range, tag_map, roles.trackers,
                                            roles.target, tracker_pose_fn, agg_fn=agg, window_len=args.init_window)
        if p0_target is not None:
            print(f"[SWARM] Initialized target p0 from window: {p0_target}", flush=True)

    target_filters = {trk: TargetIF(x0=np.hstack([p0_target if p0_target is not None else np.zeros(3),
                                                  np.zeros(3)]),
                                    cfg=tf_cfg)
                      for trk in roles.trackers}

    # Measurement adapter (bias, reliability shaping, innovation scaling)
    base_var = (args.uwb_std**2) if (args.uwb_std is not None) else args.uwb_var
    mcfg = AdapterConfig(
        base_range_var=base_var,
        los_influence=args.los_influence,
        geom_influence=args.geom_influence,
        ema_alpha=args.ema_alpha,
        min_scale=args.r_min_scale,
        max_scale=args.r_max_scale,
    )
    meas_ai = MeasureAdapter(mcfg)

    # LOS adapter
    los_adapter = LOSAdapter(LOSConfig(use_cir=args.use_cir, verbose=args.los_verbose))

    # Online self-calibration (causal, a-priori) using innovation statistics
    tuner = None
    if args.online_tune:
        tuner = OnlineTuner(OnlineAdaptConfig(
            ema_alpha=0.05,
            r_min_scale=args.online_r_min_scale,
            r_max_scale=args.online_r_max_scale,
            gate_target_accept=args.gate_target,
            gate_sigma_init=args.gate_sigma_init,
            q_adapt=args.q_adapt,
            q_alpha=0.05,
            q_gain=0.25
        ))

    # Height aligned to query timestamps (zero-order hold) for z-only target update
    height_at_q: dict = {}
    if args.use_height_tf and not height_df.empty:
        for r in robots:
            hdict = miluv.query_by_timestamps(query_timestamps, robots=r, sensors="height")[r]
            # hdict["height"] has columns [timestamp, range], bias already removed
            height_at_q[r] = hdict["height"]["range"].to_numpy(dtype=float)

    # Load BiasNet if provided
    if args.biasnet_dir is not None:
        biasnet = load_biasnet(args.biasnet_dir)
        meas_ai.bias_model = biasnet
        print(f"[SWARM] BiasNet loaded from: {args.biasnet_dir}")

    # CI fusion (auto objective if not set)
    if args.ci_objective is None:
        ci_obj = "trace" if args.use_height_tf else "logdet"
    else:
        ci_obj = args.ci_objective
    fuser = CIFuser(CIFuserConfig(objective=ci_obj, grid_step=args.ci_grid))

    # Load FusionNet if provided
    if args.fusionnet_dir is not None:
        fuser.weight_model = load_fusionnet(args.fusionnet_dir)
        print(f"[SWARM] FusionNet loaded from: {args.fusionnet_dir} (CI learned weights)")

    # Optional ML data collector (BiasNet samples + FusionNet snaps)
    collector = None
    if args.collect_bias or args.collect_fusion:
        collector = SnapshotCollector(
            query_timestamps=query_timestamps,
            roles=roles,
            tag_map=tag_map,
            gt_T_by_robot=gt_se23,
            tag_moment_arms=miluv.tag_moment_arms,
            base_var=base_var,
            pair_corr=args.pair_corr,
            huber_delta=args.huber_delta,
            los_verbose=args.los_verbose
        )

    # Storage for our target estimates
    mu_star_seq, P_star_seq = [], []

    # Counters for audit
    total_steps = len(query_timestamps)
    meas_avail, meas_used = 0, 0
    los_hits, los_misses = 0, 0

    # Vertical sensitivity logging
    vertical_sensitivities = []

    # Precompute output dir for any streaming logs we may write
    out_dir = os.path.join(args.out, f"{exp_name}_{roles.target}")

    # CI weights logging
    w_csv = None
    w_file = None
    w_header_written = False

    # NIS logging
    nis_rows = []

    # ----------------- Main loop -----------------
    for i in range(total_steps):
        t = query_timestamps[i]
        dt = (t - query_timestamps[i - 1]) if i > 0 else 0.0

        # === Authors' EKF predict ===
        u_dict = {
            r: np.array([
                gyro[r].iloc[i]["angular_velocity.x"], gyro[r].iloc[i]["angular_velocity.y"], gyro[r].iloc[i]["angular_velocity.z"],
                accel[r].iloc[i]["linear_acceleration.x"], accel[r].iloc[i]["linear_acceleration.y"], accel[r].iloc[i]["linear_acceleration.z"]
            ])
            for r in robots
        }
        ekf.predict(u_dict, dt)

        # === Authors' EKF correct: inter-robot UWB (and optional height) ===
        idx = np.where(uwb_range["timestamp"] == t)[0]
        if len(idx) > 0:
            rdata = uwb_range.iloc[idx]
            # NEW: iterate all rows at timestamp t
            for _, row in rdata.iterrows():
                ekf.correct({
                    "range": float(row["range"]),
                    "to_id": int(row["to_id"]),
                    "from_id": int(row["from_id"]),
                })
        if args.use_height and not height_df.empty:
            hidx = np.where(height_df["timestamp"] == t)[0]
            if len(hidx) > 0:
                hdata = height_df.iloc[hidx]
                # NEW: iterate all height rows (if multiple)
                for _, row in hdata.iterrows():
                    ekf.correct({
                        "height": float(row["range"]),   # column name is 'range' in height.csv
                        "robot": str(row["robot"]),
                    })

        # Store authors' EKF for post-processing (unchanged)
        for r in robots:
            ekf_history[r]["pose"].add(t, ekf.pose[r], ekf.pose_covariance[r])
            ekf_history[r]["bias"].add(t, ekf.bias[r], ekf.bias_covariance[r])

        # === Our parallel per-tracker target filtering ===
        tracker_pos = {r: se_translation_from_matrix(ekf.pose[r]) for r in roles.trackers}
        last_target_mu = mu_star_seq[-1] if len(mu_star_seq) > 0 else None
        last_target_pos = last_target_mu[:3] if last_target_mu is not None else None

        parts = {}
        node_feats = {}

        df_t = uwb_range[uwb_range["timestamp"] == t]

        for trk in roles.trackers:
            # Select only tracker↔target tag pairs (either direction)
            pair_df = select_pair_rows(df_t,
                                       trk_tags=tag_map.get(trk, []),
                                       tgt_tags=tag_map.get(roles.target, []))
            if pair_df.empty:
                # predict-only
                # Optional process-noise adaptation (global scales)
                if tuner is not None and args.q_adapt:
                    qxy_scale, qz_scale = tuner.get_q_scales()
                    target_filters[trk].cfg.sigma_a_xy = args.sigma_a_xy * float(qxy_scale)
                    target_filters[trk].cfg.sigma_a_z  = args.sigma_a_z  * float(qz_scale)
                target_filters[trk].predict(dt)
                mu_i, P_i = target_filters[trk].posterior()
                parts[trk] = (mu_i, P_i)
                # Richer node features (predict-only defaults)
                var_pos = np.trace(P_i[:3, :3])
                geom_ez = 0.0
                gate_sig = target_filters[trk].cfg.gate_N_sigma
                nis_ema  = tuner._ema_nis.get((trk, roles.target), 1.0) if (tuner is not None) else 1.0
                los_f    = 0.5
                node_feats[trk] = np.array([var_pos, 0.0, 0.0, args.uwb_var, geom_ez, los_f, gate_sig, nis_ema], dtype=float)
                continue

            # NEW: robust aggregation across all tag pairs at t
            z_agg, R_pair, mmeta = robust_range_aggregate(pair_df, base_var=base_var,
                                                          rho=args.pair_corr, huber_delta=args.huber_delta)
            meas_avail += 1

            # Choose representative tracker tag position for better geometry
            sensor_pos = robust_tracker_sensor_position(
                pair_df=pair_df,
                trk=trk,
                trk_tags=tag_map.get(trk, []),
                T_trk=ekf.pose[trk],
                tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )

            # Robust target-tag offset in world (R_tgt @ arm_tgt). If no info => zeros.
            tgt_offset_w = robust_target_offset(
                pair_df=pair_df,
                tgt_tags=tag_map.get(roles.target, []),
                T_tgt=ekf.pose[roles.target],
                tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )

            # Exact re-parameterization: shift the sensor by the target offset
            eff_sensor_pos = sensor_pos - tgt_offset_w
            z_agg_center   = float(z_agg)  # keep raw aggregated range

            feat = build_measurement_features(
                tracker_pos=eff_sensor_pos,
                target_pred_pos=last_target_pos,
                uwb_range=z_agg_center,   # <<<<<<<<<<
                los_score=None
            )

            # Get LOS score if enabled
            los_score = None
            if args.use_los:
                los_score = los_adapter.score(pair_df, extras=None)  # CIR extras can be added later
                if los_score is None:
                    los_misses += 1
                else:
                    los_hits += 1

            # Fetch per-link R scale and gate sigma from tuner (if enabled)
            link = (trk, roles.target)
            r_scale = 1.0
            if tuner is not None:
                r_scale = tuner.get_r_scale(link)
                meas_ai._rscale[link] = float(r_scale)
                target_filters[trk].cfg.gate_N_sigma = float(tuner.get_gate_sigma(link))

            # Collect BiasNet supervised sample using current geometry
            if collector and args.collect_bias:
                collector.add_bias_sample(
                    i=i,
                    trk=trk,
                    pair_df=pair_df,
                    z_agg=z_agg,
                    eff_sensor_pos_used=eff_sensor_pos,
                    los_score=los_score
                )

            z_corr, R_eff, meta = meas_ai.correct(
                tracker_id=trk, target_id=roles.target, z=z_agg_center,  # <<<<<<<<<<
                tracker_pos=eff_sensor_pos, target_pred_pos=last_target_pos,
                los_score=los_score, features=feat
            )
            # Combine adapter's reliability shaping with pair variance
            R_eff = max(R_eff, R_pair)

            # Local filter predict step (before gating)
            # Optional process-noise adaptation (global scales)
            if tuner is not None and args.q_adapt:
                qxy_scale, qz_scale = tuner.get_q_scales()
                target_filters[trk].cfg.sigma_a_xy = args.sigma_a_xy * float(qxy_scale)
                target_filters[trk].cfg.sigma_a_z  = args.sigma_a_z  * float(qz_scale)
            target_filters[trk].predict(dt)

            # Pre-gate using tuner's sigma if enabled (causal check before update)
            if tuner is not None:
                # Predict measurement at current state for gating
                h0, H = target_filters[trk]._range_linearize(target_filters[trk].mu, eff_sensor_pos)
                # Innovation and S using current covariance
                from numpy.linalg import inv
                P_pred_local = inv(target_filters[trk].J)
                # scalar-safe H P H^T + R using einsum
                S_pred = float(np.einsum('i,ij,j->', H.ravel(), P_pred_local, H.ravel()) + R_eff)
                nu_pred = float(z_corr - h0)
                gate_sigma = float(tuner.get_gate_sigma(link))
                accepted = (float((nu_pred * nu_pred) / S_pred) <= gate_sigma * gate_sigma)
                tuner.after_gating(link, accepted=accepted)
                if not accepted:
                    # Skip update if gate fails; move to posterior
                    mu_i, P_i = target_filters[trk].posterior()
                    parts[trk] = (mu_i, P_i)
                    var_pos = np.trace(P_i[:3, :3])
                    # Richer node features
                    geom_ez = 0.0
                    if last_target_pos is not None:
                        b = (last_target_pos - eff_sensor_pos) / (np.linalg.norm(last_target_pos - eff_sensor_pos) + 1e-9)
                        geom_ez = float(abs(b[2]))
                    gate_sig = tuner.get_gate_sigma(link) if tuner is not None else target_filters[trk].cfg.gate_N_sigma
                    nis_ema  = tuner._ema_nis.get(link, 1.0) if (tuner is not None) else 1.0
                    los_f    = 0.5 if (los_score is None) else float(los_score)
                    node_feats[trk] = np.array([var_pos, meta["reliability"], z_agg_center, R_eff, geom_ez, los_f, gate_sig, nis_ema], dtype=float)
                    continue
            # Optional: z-only height-difference correction
            if args.use_height_tf and height_at_q:
                if roles.target in height_at_q and trk in height_at_q:
                    # Zero-order held values aligned to query_timestamps
                    h_tgt = float(height_at_q[roles.target][i])
                    h_trk = float(height_at_q[trk][i])
                    if np.isfinite(h_tgt) and np.isfinite(h_trk):
                        dz_meas = h_tgt - h_trk
                        R_h = 2.0 * (args.height_std ** 2)
                        # Use robot center/world z for the height sensor, not UWB tag geometry
                        z_trk = float(tracker_pos[trk][2])
                        target_filters[trk].correct_height(dz_meas, z_trk, R_h)
            upd = target_filters[trk].correct(z_corr, R_eff, tracker_pos=eff_sensor_pos)
            # No extra tuner.after_gating() here to avoid double-counting
            if upd.get("used", False):
                meas_used += 1
                # NEW: feed innovation and S back to the adapter for future steps
                meas_ai.update_from_innov(tracker_id=trk, target_id=roles.target,
                                          innov=upd.get("innov", None), S=upd.get("S", None))

                # Log vertical sensitivity for observability analysis and feed tuner
                geom_ez_val = None
                if last_target_pos is not None:
                    bearing = (last_target_pos - eff_sensor_pos) / np.linalg.norm(last_target_pos - eff_sensor_pos)
                    geom_ez_val = float(abs(bearing[2]))
                    vertical_sensitivities.append(geom_ez_val)  # |e_z|
                # Feed per-link innovation stats back to tuner (causal)
                if tuner is not None:
                    tuner.after_update(
                        link=link,
                        innovation=float(upd.get("innov", 0.0)),
                        S_scalar=float(upd.get("S", 1.0)),
                        r_is_maxed=bool(float(r_scale) >= 0.6 * float(tuner.cfg.r_max_scale)),
                        geom_ez=geom_ez_val
                    )
                # Log NIS
                if upd.get("S", None) is not None and upd.get("innov", None) is not None:
                    nis_rows.append([float(t), trk, float((upd['innov']**2)/upd['S']), float(upd['S'])])

            mu_i, P_i = target_filters[trk].posterior()
            parts[trk] = (mu_i, P_i)
            var_pos = np.trace(P_i[:3, :3])
            # keep features consistent with the actual measurement and geometry
            geom_ez = 0.0
            if last_target_pos is not None:
                b = (last_target_pos - eff_sensor_pos) / (np.linalg.norm(last_target_pos - eff_sensor_pos) + 1e-9)
                geom_ez = float(abs(b[2]))
            gate_sig = tuner.get_gate_sigma(link) if tuner is not None else target_filters[trk].cfg.gate_N_sigma
            nis_ema  = tuner._ema_nis.get(link, 1.0) if (tuner is not None) else 1.0
            los_f    = 0.5 if (los_score is None) else float(los_score)
            node_feats[trk] = np.array([var_pos, meta["reliability"], z_agg_center, R_eff, geom_ez, los_f, gate_sig, nis_ema], dtype=float)

        # Collect FusionNet snapshot before fusion (node features + local posteriors)
        if collector and args.collect_fusion:
            collector.add_fusion_snap(i=i, parts=parts, node_feats=node_feats)

        # CI fuse per-tracker posteriors
        mu_star, P_star, w = fuser.fuse(parts, method=args.ci_method,
                                        node_features={k: node_feats[k] for k in parts.keys()})
        mu_star_seq.append(mu_star)
        P_star_seq.append(P_star)
        # Log CI weights
        try:
            if w_file is None:
                import csv
                os.makedirs(out_dir, exist_ok=True)
                w_file = open(os.path.join(out_dir, "fusion_weights.csv"), "w", newline="")
                w_csv = csv.writer(w_file)
                # Use current order of keys
                w_csv.writerow(["timestamp"] + [f"w_{rid}" for rid in w.keys()])
            w_csv.writerow([float(t)] + [float(w[rid]) for rid in w.keys()])
        except Exception:
            pass

    # ----------------- Optional RTS smoothing -----------------
    if args.smooth:
        print("[SWARM] Applying RTS smoothing...")
        dts = np.diff(query_timestamps, prepend=query_timestamps[0])
        # Heuristic: smoother noise >= filter noise to avoid overconfidence
        sax = max(args.smooth_sigma_a_xy, args.sigma_a_xy)
        saz = max(args.smooth_sigma_a_z,  args.sigma_a_z)
        mu_star_seq, P_star_seq = rts_smooth(
            mu_star_seq, P_star_seq, dts, CVNoise(sigma_a_xy=sax, sigma_a_z=saz)
        )

    # ----------------- Evaluate target tracking -----------------
    gt_tgt_pos = np.array([se_translation_from_matrix(T) for T in gt_se23[roles.target]])
    rm, nees_val = evaluate_and_save(np.array(mu_star_seq), np.array(P_star_seq), gt_tgt_pos, out_dir)

    with open(os.path.join(out_dir, "roles.json"), "w") as f:
        json.dump({"target": roles.target, "trackers": roles.trackers, "tag_map": tag_map}, f, indent=2)

    # Save vertical sensitivity data for observability analysis
    if vertical_sensitivities:
        with open(os.path.join(out_dir, "vertical_sensitivity.json"), "w") as f:
            json.dump({
                "vertical_sensitivities": vertical_sensitivities,
                "mean_sensitivity": float(np.mean(vertical_sensitivities)),
                "median_sensitivity": float(np.median(vertical_sensitivities)),
                "min_sensitivity": float(np.min(vertical_sensitivities)),
                "max_sensitivity": float(np.max(vertical_sensitivities))
            }, f, indent=2)

    # Save trajectory and basic diagnostics (separate file to avoid overwriting evaluate_and_save output)
    out_csv = os.path.join(out_dir, "target_state.csv")
    ts = np.asarray(query_timestamps).reshape(-1, 1)
    X = np.asarray(mu_star_seq)  # N x 6 [px,py,pz,vx,vy,vz]
    df = pd.DataFrame(np.hstack([ts, X]),
                      columns=["timestamp","px","py","pz","vx","vy","vz"]) 
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] Trajectory -> {out_csv}")

    # Persist NIS log if available
    if nis_rows:
        import csv
        with open(os.path.join(out_dir, "nis_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","tracker","nis","S"])
            w.writerows(nis_rows)

    # Close weights CSV if open
    if w_file is not None:
        try:
            w_file.close()
        except Exception:
            pass

    # Persist collected ML data if enabled
    if collector:
        coll_dir = args.collect_dir if args.collect_dir else out_dir
        collector.save(coll_dir)

    print(f"[SWARM] Measurements available (trk↔tgt) : {meas_avail}")
    print(f"[SWARM] Measurements used after gating : {meas_used}")
    print(f"[SWARM] LOS scores produced: {los_hits}, missing: {los_misses}")
    if vertical_sensitivities:
        print(f"[SWARM] Vertical sensitivity: mean={np.mean(vertical_sensitivities):.3f}, median={np.median(vertical_sensitivities):.3f}")
    print(f"[DONE] Results written to: {out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--exp", required=True, help="Experiment name, e.g., default_3_random_0")
    p.add_argument("--target", default=None, help="Robot id to treat as target (default: last in sort)")
    p.add_argument("--use_height", action="store_true", help="Include height correction in authors' EKF")
    p.add_argument("--use_height_tf", action="store_true", help="Use PX4 height to update the target filter (z-only)")
    p.add_argument("--height_std", type=float, default=0.07, help="Std dev (m) of PX4 height per sensor; used for R_h")
    p.add_argument("--sigma_a", type=float, default=1.0, help="Target process accel noise std (m/s^2) [legacy, use sigma_a_xy/z]")
    p.add_argument("--sigma_a_xy", type=float, default=1.0, help="Horiz accel noise std (m/s^2)")
    p.add_argument("--sigma_a_z", type=float, default=0.5, help="Vertical accel noise std (m/s^2)")
    p.add_argument("--p0", type=float, default=2.0, help="Initial target position std (m)")
    p.add_argument("--v0", type=float, default=1.0, help="Initial target velocity std (m/s)")
    # More conservative default UWB variance (std 0.35 m)
    p.add_argument("--uwb_var", type=float, default=0.35**2, help="Baseline UWB variance (m^2)")
    p.add_argument("--uwb_std", type=float, default=None, help="Alternative: give UWB std (m); overrides --uwb_var")
    p.add_argument("--gate_sigma", type=float, default=3.0, help="Gating threshold in sigma")
    p.add_argument("--ci_method", choices=["uniform", "grid", "learned"], default="grid")
    p.add_argument("--ci_objective", choices=["logdet","trace"], default=None)
    p.add_argument("--ci_grid", type=float, default=0.1, help="Grid step for CI weights")
    p.add_argument("--tags_per_robot", type=int, default=2, help="How many tags to assume per robot")
    p.add_argument("--pair_corr", type=float, default=0.7, help="Correlation between tag-pair ranges")
    p.add_argument("--huber_delta", type=float, default=0.8, help="Huber delta for per-timestep range aggregation (m)")
    p.add_argument("--use_los", action="store_true", help="Use LOS classifier to shape reliability")
    p.add_argument("--use_cir", action="store_true", help="If available, enable CIR for LOS classifier")
    p.add_argument("--los_verbose", action="store_true", help="Print one-time LOS adapter diagnostics")
    p.add_argument("--los_influence", type=float, default=0.2, help="Strength of LOS->reliability (0..1)")
    p.add_argument("--geom_influence", type=float, default=0.4, help="Strength of |e_z|->reliability (0..1)")
    # IMPORTANT: when --online_tune is ON, set --ema_alpha 0.0 to avoid double R-scaling
    p.add_argument("--ema_alpha", type=float, default=0.0, help="EMA for innovation whiteness (MeasureAdapter)")
    p.add_argument("--r_min_scale", type=float, default=0.5, help="Lower bound on R scaling")
    p.add_argument("--r_max_scale", type=float, default=6.0, help="Upper bound on R scaling")
    # Online tuner flags
    p.add_argument("--online_tune", action="store_true", help="Enable OnlineTuner (causal self-calibration)")
    p.add_argument("--online_r_min_scale", type=float, default=0.75)
    p.add_argument("--online_r_max_scale", type=float, default=10.0)
    p.add_argument("--gate_target", type=float, default=0.97)
    p.add_argument("--gate_sigma_init", type=float, default=3.0)
    p.add_argument("--q_adapt", action="store_true", help="Let tuner gently scale process noise (Q)")
    p.add_argument("--init_window", type=int, default=0,
                   help="Use first N timesteps to robustly initialize target position (0=off)")
    p.add_argument("--smooth", action="store_true", help="Enable RTS smoothing after filtering")
    p.add_argument("--smooth_sigma_a_xy", type=float, default=1.0, help="RTS smoother horizontal accel noise std (m/s^2)")
    p.add_argument("--smooth_sigma_a_z", type=float, default=0.7, help="RTS smoother vertical accel noise std (m/s^2)")
    p.add_argument("--biasnet_dir", default=None, help="Directory containing biasnet.pt and biasnet_meta.json")
    p.add_argument("--fusionnet_dir", default=None, help="Directory containing fusionnet.pt and fusionnet_meta.json")
    p.add_argument("--out", default="outputs_swarm")
    p.add_argument("--collect_bias", action="store_true", help="Collect BiasNet samples during run")
    p.add_argument("--collect_fusion", action="store_true", help="Collect FusionNet snaps during run")
    p.add_argument("--collect_dir", default=None, help="Override output dir for collected data (default: experiment out dir)")
    args = p.parse_args()
    main(args)
