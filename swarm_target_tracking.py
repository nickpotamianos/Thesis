# swarm_target_tracking.py
import argparse, os, json
import numpy as np
import pandas as pd

def load_fusionnet(path_dir: str):
    import os, json, torch
    from swarm_ml.models import FusionNet
    with open(os.path.join(path_dir, "fusionnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    x_mu = meta.get("x_mu", None)
    x_std = meta.get("x_std", None)
    
    model = FusionNet(in_dim)
    # weights_only=True to address the FutureWarning
    state = torch.load(os.path.join(path_dir, "fusionnet.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=False)
    
    # Reinstall normalizer (stored during training by train_fusionnet.py)
    if x_mu is not None and x_std is not None:
        model.set_normalizer(x_mu, x_std)
        print(f"[SWARM] FusionNet normalizer restored: {len(x_mu)} features")
    else:
        print(f"[SWARM] Warning: No normalizer data found in meta.json")
    
    model.eval()
    return model

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
from swarm_ml.distrib_ci import GossipFuser, CommsConfig
from swarm_ml.planning import suggest_vantage_moves, suggest_vantage_moves_eig
from swarm_ml.active_sensing import expected_trace_reduction
from swarm_control.bridge import ControlBridge

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
        if isinstance(tag_moment_arms, dict):
            if tag_id in tag_moment_arms:
                arm = np.asarray(tag_moment_arms[tag_id], dtype=float).reshape(3)
            else:
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
    state = torch.load(os.path.join(path_dir, "biasnet.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    # Restore normalizer if present
    try:
        x_mu  = meta.get("x_mu", None)
        x_std = meta.get("x_std", None)
        if x_mu is not None and x_std is not None:
            model.set_normalizer(x_mu, x_std)
            print(f"[SWARM] BiasNet normalizer restored: {len(x_mu)} features")
    except Exception as e:
        print(f"[SWARM] Warning: could not set BiasNet normalizer: {e}")
    model.eval()
    return model

def load_fusionnet_legacy(path_dir: str):
    import json, torch
    from swarm_ml.models import FusionNet
    with open(os.path.join(path_dir, "fusionnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    model = FusionNet(in_dim)
    state = torch.load(os.path.join(path_dir, "fusionnet.pt"), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model

def main(args):
    exp_name = args.exp
    os.makedirs(args.out, exist_ok=True)

    # ----------------- Load data (authors' devkit) -----------------
    miluv = DataLoader(
        exp_name,
        exp_dir="./data/three_robots",
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

    # Per-tracker target filters
    tf_cfg = IFConfig(
        sigma_a_xy=args.sigma_a_xy,
        sigma_a_z=args.sigma_a_z,
        p0_xy=args.p0, p0_z=args.p0,
        v0_xy=args.v0, v0_z=args.v0,
        gate_N_sigma=args.gate_sigma
    )

    # Optional: robust target initialization from first window
    from swarm_ml.target_init import init_target_from_window
    from swarm_ml.tagmap import robust_range_aggregate
    p0_target = None
    if args.init_window > 0:
        def tracker_pose_fn(trk, t):
            T_trk = ekf.pose.get(trk, None)
            if T_trk is None:
                return np.zeros(3)
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
        bias_model_gain=args.bias_gain,
        ema_alpha=args.ema_alpha,
        min_scale=args.r_min_scale,
        max_scale=args.r_max_scale,
        # If OnlineTuner is enabled, let it own R scaling.
        own_rscale=(not args.online_tune),
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
            height_at_q[r] = hdict["height"]["range"].to_numpy(dtype=float)

    # Load BiasNet if provided
    if args.biasnet_dir is not None:
        biasnet = load_biasnet(args.biasnet_dir)
        meas_ai.bias_model = biasnet
        print(f"[SWARM] BiasNet loaded from: {args.biasnet_dir}")

    # CI fusion & (optional) learned weights
    if args.ci_objective is None:
        ci_obj = "trace" if args.use_height_tf else "logdet"
    else:
        ci_obj = args.ci_objective
    fuser = CIFuser(CIFuserConfig(objective=ci_obj, grid_step=args.ci_grid))

    if args.fusionnet_dir is not None:
        fuser.weight_model = load_fusionnet(args.fusionnet_dir)
        print(f"[SWARM] FusionNet loaded from: {args.fusionnet_dir} (CI learned weights)")

    # Gossip fuser for decentralized CI
    gossip = None
    if args.decentralized:
        gossip = GossipFuser(CommsConfig(
            rounds=args.comm_rounds, p_link=args.comm_p, p_drop=args.comm_drop, seed=args.comm_seed))
        print(f"[SWARM] Decentralized gossip CI enabled: p_link={args.comm_p}, p_drop={args.comm_drop}, rounds={args.comm_rounds}, seed={args.comm_seed}")

    # Control bridge (optional live publish)
    ctrl = ControlBridge(mode=args.control_mode, rate_hz=args.control_rate)

    # Optional ML data collector (BiasNet samples + FusionNet snaps)
    collector = None
    if args.collect_bias or args.collect_fusion:
        collector = SnapshotCollector(
            exp_name=exp_name,
            query_timestamps=query_timestamps,
            roles=roles,
            tag_map=tag_map,
            gt_T_by_robot=gt_se23,
            tag_moment_arms=miluv.tag_moment_arms,
            base_var=base_var,
            pair_corr=args.pair_corr,
            huber_delta=args.huber_delta,
            los_verbose=args.los_verbose,
            height_series=(height_at_q if args.use_height_tf else None)
        )

    mu_star_seq, P_star_seq = [], []
    total_steps = len(query_timestamps)
    meas_avail, meas_used = 0, 0
    los_hits, los_misses = 0, 0
    vertical_sensitivities = []
    action_rows = []  # For action suggestions CSV

    out_dir = os.path.join(args.out, f"{exp_name}_{roles.target}")

    if args.collect_bias or args.collect_fusion:
        print(f"[COLLECT] Enabled -> bias={args.collect_bias}, fusion={args.collect_fusion}")
        print(f"[COLLECT] Output directory: {out_dir}")


    w_csv = None; w_file = None
    nis_rows = []
    eig_scores_rows = []

    # ----------------- Main loop -----------------
    for i in range(total_steps):
        t = query_timestamps[i]
        dt = (t - query_timestamps[i - 1]) if i > 0 else 0.0

        # --- Authors' EKF predict ---
        u_dict = {
            r: np.array([
                gyro[r].iloc[i]["angular_velocity.x"], gyro[r].iloc[i]["angular_velocity.y"], gyro[r].iloc[i]["angular_velocity.z"],
                accel[r].iloc[i]["linear_acceleration.x"], accel[r].iloc[i]["linear_acceleration.y"], accel[r].iloc[i]["linear_acceleration.z"]
            ])
            for r in robots
        }
        ekf.predict(u_dict, dt)

        # --- Authors' EKF correct ---
        idx = np.where(uwb_range["timestamp"] == t)[0]
        if len(idx) > 0:
            rdata = uwb_range.iloc[idx]
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
                for _, row in hdata.iterrows():
                    ekf.correct({
                        "height": float(row["range"]),
                        "robot": str(row["robot"]),
                    })

        # Store EKF state history for authors' devkit
        for r in robots:
            ekf_history[r]["pose"].add(t, ekf.pose[r], ekf.pose_covariance[r])
            ekf_history[r]["bias"].add(t, ekf.bias[r], ekf.bias_covariance[r])

        # --- Our parallel per-tracker target filtering ---
        tracker_pos = {r: se_translation_from_matrix(ekf.pose[r]) for r in roles.trackers}
        last_target_mu = mu_star_seq[-1] if len(mu_star_seq) > 0 else None
        last_target_pos = last_target_mu[:3] if last_target_mu is not None else None

        parts = {}
        node_feats = {}
        df_t = uwb_range[uwb_range["timestamp"] == t]

        info_gain_map = {}
        r_eff_map = {}
        for trk in roles.trackers:
            # Select only tracker↔target tag pairs (either direction)
            pair_df = select_pair_rows(df_t,
                                       trk_tags=tag_map.get(trk, []),
                                       tgt_tags=tag_map.get(roles.target, []))
            if pair_df.empty:
                # predict-only
                if tuner is not None and args.q_adapt:
                    qxy_scale, qz_scale = tuner.get_q_scales()
                    target_filters[trk].cfg.sigma_a_xy = args.sigma_a_xy * float(qxy_scale)
                    target_filters[trk].cfg.sigma_a_z  = args.sigma_a_z  * float(qz_scale)
                target_filters[trk].predict(dt)
                mu_i, P_i = target_filters[trk].posterior()
                parts[trk] = (mu_i, P_i)
                var_pos = np.trace(P_i[:3, :3])
                gate_sig = target_filters[trk].cfg.gate_N_sigma
                nis_ema  = tuner._ema_nis.get((trk, roles.target), 1.0) if (tuner is not None) else 1.0
                los_f    = 0.5
                # Consistent 11-dimensional features (fallback values for missing pairs)
                node_feats[trk] = np.array([
                    var_pos,                # 0
                    0.0,                    # 1 - reliability
                    0.0,                    # 2 - z_agg_center  
                    args.uwb_var,           # 3 - R_eff
                    0.0,                    # 4 - geom_ez
                    los_f,                  # 5 - los_score
                    gate_sig,               # 6 - gate_sigma
                    nis_ema,                # 7 - nis_ema
                    args.uwb_var,           # 8 - R_pair (fallback)
                    1.0,                    # 9 - m_eff (single fallback)
                    0.0                     # 10 - iqr (no spread)
                ], dtype=float)
                info_gain_map[trk] = 0.0
                continue

            # robust aggregation across all tag pairs at t
            z_agg, R_pair, meta_pairs = robust_range_aggregate(pair_df, base_var=base_var,
                                                      rho=args.pair_corr, huber_delta=args.huber_delta)
            meas_avail += 1

            # tracker sensor world pos and target-tag offset
            sensor_pos = robust_tracker_sensor_position(
                pair_df=pair_df,
                trk=trk,
                trk_tags=tag_map.get(trk, []),
                T_trk=ekf.pose[trk],
                tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )
            tgt_offset_w = robust_target_offset(
                pair_df=pair_df,
                tgt_tags=tag_map.get(roles.target, []),
                T_tgt=ekf.pose[roles.target],
                tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )
            eff_sensor_pos = sensor_pos - tgt_offset_w
            z_agg_center   = float(z_agg)

            # Optional: build a time-windowed set just for LOS/IQR
            pair_df_los = pair_df
            if args.los_window and args.los_window > 0.0:
                t0, t1 = float(t - args.los_window), float(t + args.los_window)
                df_win = uwb_range[(uwb_range["timestamp"] >= t0) &
                                   (uwb_range["timestamp"] <= t1) &
                                   (uwb_range["robot"] == trk)]
                pair_df_los = select_pair_rows(
                    df_win,
                    trk_tags=tag_map.get(trk, []),
                    tgt_tags=tag_map.get(roles.target, [])
                )

            # LOS score if enabled (compute BEFORE features to maintain train‑test parity)
            los_score = None
            if args.use_los:
                # use the windowed set so m≥3 is common
                los_score = los_adapter.score(pair_df_los, extras=None)
                if los_score is None:
                    los_misses += 1
                else:
                    los_hits += 1

            # Optional heights for feature parity (ignored if None)
            h_trk = None
            h_tgt = None
            if args.use_height_tf and height_at_q:
                try:
                    if trk in height_at_q:
                        h_trk = float(height_at_q[trk][i])
                    if roles.target in height_at_q:
                        h_tgt = float(height_at_q[roles.target][i])
                except Exception:
                    h_trk = None; h_tgt = None

            # Feature vector (must match training layout)
            feat = build_measurement_features(
                tracker_pos=eff_sensor_pos,
                target_pred_pos=None,          # match training (no geometry inputs)
                uwb_range=z_agg_center,
                los_score=los_score if args.use_los else None,
                height_tracker=(float(height_at_q[trk][i])             if (args.use_height_tf and height_at_q and (trk in height_at_q)) else None),
                height_target=(float(height_at_q[roles.target][i])     if (args.use_height_tf and height_at_q and (roles.target in height_at_q)) else None)
            )
            
            # --- Extend with pair-quality & identity (must match collector) ---
            m_eff = float(meta_pairs.get("m_eff", 1.0))
            # IQR from the windowed set if present, else meta / safe fallback
            if pair_df_los is not None and not pair_df_los.empty:
                zs_los = pair_df_los["range"].to_numpy(dtype=float)
                if zs_los.size >= 3:
                    q25, q75 = np.percentile(zs_los, [25, 75]); iqr = float(max(0.0, q75 - q25))
                elif zs_los.size == 2:
                    iqr = float(abs(zs_los[1] - zs_los[0]))
                else:
                    iqr = float(meta_pairs.get("iqr", 0.0))
            else:
                iqr = float(meta_pairs.get("iqr", 0.0))
            tracker_vocab = sorted(list(roles.trackers))
            onehot = np.zeros(len(tracker_vocab), dtype=float)
            onehot[tracker_vocab.index(trk)] = 1.0
            feat = np.hstack([feat, [float(R_pair), m_eff, iqr], onehot])

            # Per-link R scaling + gating from tuner
            link = (trk, roles.target)
            r_scale = 1.0
            if tuner is not None:
                r_scale = tuner.get_r_scale(link)
                meas_ai._rscale[link] = float(r_scale)
                target_filters[trk].cfg.gate_N_sigma = float(tuner.get_gate_sigma(link))

            # Collect BiasNet sample (optional)
            collector and args.collect_bias and collector.add_bias_sample(
                i=i, trk=trk, pair_df=pair_df, z_agg=z_agg,
                eff_sensor_pos_used=eff_sensor_pos, los_score=los_score
            )

            # Adapter correction (bias + reliability)
            z_corr, R_eff, meta = meas_ai.correct(
                tracker_id=trk, target_id=roles.target, z=z_agg_center,
                tracker_pos=eff_sensor_pos, target_pred_pos=last_target_pos,
                los_score=los_score, features=feat
            )
            # Soft R-floor blending to avoid hard lower-bounding
            if True:
                # blend factor from CLI
                k = float(getattr(args, 'r_floor_blend', 0.5))
                R_eff = k * float(R_eff) + (1.0 - k) * float(max(R_eff, R_pair))
            else:
                R_eff = max(R_eff, R_pair)
            r_eff_map[trk] = float(R_eff)

            # Local filter predict step (before gating)
            if tuner is not None and args.q_adapt:
                qxy_scale, qz_scale = tuner.get_q_scales()
                target_filters[trk].cfg.sigma_a_xy = args.sigma_a_xy * float(qxy_scale)
                target_filters[trk].cfg.sigma_a_z  = args.sigma_a_z  * float(qz_scale)
            target_filters[trk].predict(dt)

            # Compute approximate one-step information gain (trace reduction) for budgeting
            try:
                h0_tmp, H_tmp = target_filters[trk]._range_linearize(target_filters[trk].mu, eff_sensor_pos)
                from numpy.linalg import inv
                P_pred_local_for_gain = inv(target_filters[trk].J)
                Hc = H_tmp.reshape(-1, 1)
                J_add = (1.0 / float(R_eff)) * (Hc @ Hc.T)
                # A-optimal trace reduction
                gain_trace = float(np.trace(P_pred_local_for_gain) - np.trace(inv(target_filters[trk].J + J_add)))
            except Exception:
                gain_trace = 0.0
            info_gain_map[trk] = max(0.0, gain_trace)

            # Pre-gate (causal) using tuner's threshold if enabled
            if tuner is not None:
                h0, H = target_filters[trk]._range_linearize(target_filters[trk].mu, eff_sensor_pos)
                from numpy.linalg import inv
                P_pred_local = inv(target_filters[trk].J)
                S_pred = float(np.einsum('i,ij,j->', H.ravel(), P_pred_local, H.ravel()) + R_eff)
                nu_pred = float(z_corr - h0)
                gate_sigma = float(tuner.get_gate_sigma(link))
                accepted = (float((nu_pred * nu_pred) / S_pred) <= gate_sigma * gate_sigma)
                tuner.after_gating(link, accepted=accepted)
                if not accepted:
                    info_gain_map[trk] = 0.0
                    mu_i, P_i = target_filters[trk].posterior()
                    parts[trk] = (mu_i, P_i)
                    var_pos = np.trace(P_i[:3, :3])
                    geom_ez = 0.0
                    if last_target_pos is not None:
                        b = (last_target_pos - eff_sensor_pos) / (np.linalg.norm(last_target_pos - eff_sensor_pos) + 1e-9)
                        geom_ez = float(abs(b[2]))
                    gate_sig = tuner.get_gate_sigma(link) if tuner is not None else target_filters[trk].cfg.gate_N_sigma
                    nis_ema  = tuner._ema_nis.get(link, 1.0) if (tuner is not None) else 1.0
                    los_f    = 0.5 if (los_score is None) else float(los_score)
                    # Enhanced features: add tag-pair stats for better discrimination
                    m_eff = float(meta_pairs.get("m_eff", 1.0))   # effective pairs
                    # Prefer IQR from the windowed set if available; fall back to per‑t meta
                    if pair_df_los is not None and not pair_df_los.empty and pair_df_los.shape[0] >= 3:
                        zs_los = pair_df_los["range"].to_numpy(dtype=float)
                        q25, q75 = np.percentile(zs_los, [25, 75])
                        iqr = float(max(0.0, q75 - q25))
                    else:
                        iqr = float(meta_pairs.get("iqr", 0.0))     # range IQR
                    node_feats[trk] = np.array([
                        var_pos,                 # 0
                        meta["reliability"],     # 1
                        z_agg_center,            # 2
                        R_eff,                   # 3
                        geom_ez,                 # 4
                        los_f,                   # 5
                        gate_sig,                # 6
                        nis_ema,                 # 7
                        R_pair,                  # 8  <- new: pair variance
                        m_eff,                   # 9  <- new: effective pairs
                        iqr                      # 10 <- new: range IQR
                    ], dtype=float)
                    continue

            # Optional: z-only height-difference correction
            if args.use_height_tf and height_at_q:
                if roles.target in height_at_q and trk in height_at_q:
                    h_tgt = float(height_at_q[roles.target][i])
                    h_trk = float(height_at_q[trk][i])
                    if np.isfinite(h_tgt) and np.isfinite(h_trk):
                        dz_meas = h_tgt - h_trk
                        R_h = 2.0 * (args.height_std ** 2)
                        z_trk = float(tracker_pos[trk][2])
                        target_filters[trk].correct_height(dz_meas, z_trk, R_h)

            upd = target_filters[trk].correct(z_corr, R_eff, tracker_pos=eff_sensor_pos)
            if upd.get("used", False):
                meas_used += 1
                meas_ai.update_from_innov(tracker_id=trk, target_id=roles.target,
                                          innov=upd.get("innov", None), S=upd.get("S", None))
                geom_ez_val = None
                if last_target_pos is not None:
                    bearing = (last_target_pos - eff_sensor_pos) / np.linalg.norm(last_target_pos - eff_sensor_pos)
                    geom_ez_val = float(abs(bearing[2]))
                    vertical_sensitivities.append(geom_ez_val)
                if tuner is not None:
                    tuner.after_update(
                        link=link,
                        innovation=float(upd.get("innov", 0.0)),
                        S_scalar=float(upd.get("S", 1.0)),
                        r_is_maxed=bool(float(r_scale) >= 0.6 * float(tuner.cfg.r_max_scale)),
                        geom_ez=geom_ez_val
                    )
                if upd.get("S", None) is not None and upd.get("innov", None) is not None:
                    nis_rows.append([float(t), trk, float((upd['innov']**2)/upd['S']), float(upd['S']), float(R_eff)])

            mu_i, P_i = target_filters[trk].posterior()
            parts[trk] = (mu_i, P_i)
            var_pos = np.trace(P_i[:3, :3])
            geom_ez = 0.0
            if last_target_pos is not None:
                b = (last_target_pos - eff_sensor_pos) / (np.linalg.norm(last_target_pos - eff_sensor_pos) + 1e-9)
                geom_ez = float(abs(b[2]))
            gate_sig = tuner.get_gate_sigma(link) if tuner is not None else target_filters[trk].cfg.gate_N_sigma
            nis_ema  = tuner._ema_nis.get(link, 1.0) if (tuner is not None) else 1.0
            los_f    = 0.5 if (los_score is None) else float(los_score)
            # Enhanced features: add tag-pair stats for better discrimination
            m_eff = float(meta_pairs.get("m_eff", 1.0))   # effective pairs
            # Prefer IQR from the windowed set if available; fall back to per‑t meta
            if pair_df_los is not None and not pair_df_los.empty and pair_df_los.shape[0] >= 3:
                zs_los = pair_df_los["range"].to_numpy(dtype=float)
                q25, q75 = np.percentile(zs_los, [25, 75])
                iqr = float(max(0.0, q75 - q25))
            else:
                iqr = float(meta_pairs.get("iqr", 0.0))     # range IQR
            node_feats[trk] = np.array([
                var_pos,                 # 0
                meta["reliability"],     # 1
                z_agg_center,            # 2
                R_eff,                   # 3
                geom_ez,                 # 4
                los_f,                   # 5
                gate_sig,                # 6
                nis_ema,                 # 7
                R_pair,                  # 8  <- new: pair variance
                m_eff,                   # 9  <- new: effective pairs
                iqr                      # 10 <- new: range IQR
            ], dtype=float)

        # Collect FusionNet snapshot before fusion (node features + local posteriors)
        if collector and args.collect_fusion:
            collector.add_fusion_snap(i=i, parts=parts, node_feats=node_feats)

        # Decide which trackers to keep under budget
        keep_keys = list(parts.keys())
        if args.budget_k is not None and len(keep_keys) > args.budget_k:
            # Prefer information-gain (A-opt trace reduction); fallback to weights or reliability
            try:
                scored = [(k, float(info_gain_map.get(k, 0.0))) for k in keep_keys]
                keep_keys = [k for k,_ in sorted(scored, key=lambda x: -x[1])[:args.budget_k]]
            except Exception:
                if fuser.weight_model is not None:
                    X_stack = np.vstack([node_feats[k].reshape(1, -1) for k in keep_keys])
                    w_pred = fuser.weight_model.predict_weights(X_stack)  # (N,)
                    order = np.argsort(-w_pred)[:args.budget_k]
                    keep_keys = [keep_keys[i] for i in order]
                else:
                    scored = [(k, float(node_feats[k][1])) for k in keep_keys]
                    keep_keys = [k for k,_ in sorted(scored, key=lambda x: -x[1])[:args.budget_k]]

        # Reduce to budgeted set
        parts = {k: parts[k] for k in keep_keys}
        node_feats = {k: node_feats[k] for k in keep_keys}

        # CI fuse per-tracker posteriors (centralized or decentralized)
        if args.decentralized:
            if fuser.weight_model is not None and len(parts) > 0:
                X_stack = np.vstack([node_feats[k].reshape(1, -1) for k in parts.keys()])
                w_vec = fuser.weight_model.predict_weights(X_stack)
                w_map = {k: float(w_vec[i]) for i, k in enumerate(parts.keys())}
                mu_star, P_star, w = gossip.fuse(parts, weights=w_map)
            else:
                mu_star, P_star, w = gossip.fuse(parts)
        else:
            mu_star, P_star, w = fuser.fuse(parts, method=args.ci_method,
                                            node_features={k: node_feats[k] for k in parts.keys()})
        mu_star_seq.append(mu_star)
        P_star_seq.append(P_star)

        # Generate action suggestions for active sensing
        if len(parts) > 0:
            if args.planner == 'eig':
                moves = suggest_vantage_moves_eig(mu_star, P_star, tracker_pos, r_eff_map)
            else:
                moves = suggest_vantage_moves(mu_star[:3], tracker_pos)
            for trk, mv in moves.items():
                action_rows.append([float(t), trk, float(mv[0]), float(mv[1]), float(mv[2])])
            # EIG diagnostics for chosen move
            if args.planner == 'eig':
                for trk, mv in moves.items():
                    try:
                        p = tracker_pos[trk].reshape(3)
                        cand_p = p + np.asarray(mv, float).reshape(3)
                        eig_val = expected_trace_reduction(mu_star, P_star, cand_p, r_eff_map.get(trk, args.uwb_var))
                        eig_scores_rows.append([float(t), trk, float(eig_val), float(r_eff_map.get(trk, args.uwb_var))])
                    except Exception:
                        pass
            # Optional live publish (sim/mavsdk)
            try:
                ctrl.send_vantage_moves(float(t), moves)
            except Exception:
                pass

        # Log CI weights to CSV
        try:
            if w_file is None:
                import csv
                os.makedirs(out_dir, exist_ok=True)
                w_file = open(os.path.join(out_dir, "fusion_weights.csv"), "w", newline="")
                w_csv = csv.writer(w_file)
                # Stable header across the whole run: all declared trackers
                header = ["timestamp"] + [f"w_{rid}" for rid in roles.trackers]
                w_csv.writerow(header)
            # Write a stable row in the same order; NaN if not present in this fuse
            row = [float(t)] + [float(w.get(rid, float('nan'))) for rid in roles.trackers]
            w_csv.writerow(row)
        except Exception:
            pass

    # ----------------- Optional RTS smoothing -----------------
    if args.smooth:
        print("[SWARM] Applying RTS smoothing...")
        dts = np.diff(query_timestamps, prepend=query_timestamps[0])
        sax = max(args.smooth_sigma_a_xy, args.sigma_a_xy)
        saz = max(args.smooth_sigma_a_z,  args.sigma_a_z)
        mu_star_seq, P_star_seq = rts_smooth(
            mu_star_seq, P_star_seq, dts, CVNoise(sigma_a_xy=sax, sigma_a_z=saz)
        )

    # ----------------- Evaluate target tracking -----------------
    gt_tgt_pos = np.array([se_translation_from_matrix(T) for T in gt_se23[roles.target]])
    rm, nees_val = evaluate_and_save(np.array(mu_star_seq), np.array(P_star_seq), gt_tgt_pos, out_dir)

    with open(os.path.join(out_dir, "roles.json"), "w") as f:
        json.dump({
            "target": roles.target,
            "trackers": roles.trackers,
            "tag_map": tag_map,
            "comm": {"p_link": args.comm_p, "p_drop": args.comm_drop, "rounds": args.comm_rounds, "seed": args.comm_seed}
        }, f, indent=2)

    # Save vertical sensitivity data
    if vertical_sensitivities:
        with open(os.path.join(out_dir, "vertical_sensitivity.json"), "w") as f:
            json.dump({
                "vertical_sensitivities": vertical_sensitivities,
                "mean_sensitivity": float(np.mean(vertical_sensitivities)),
                "median_sensitivity": float(np.median(vertical_sensitivities)),
                "min_sensitivity": float(np.min(vertical_sensitivities)),
                "max_sensitivity": float(np.max(vertical_sensitivities))
            }, f, indent=2)

    # Save trajectory and basic diagnostics
    out_csv = os.path.join(out_dir, "target_state.csv")
    ts = np.asarray(query_timestamps).reshape(-1, 1)
    X = np.asarray(mu_star_seq)  # N x 6 [px,py,pz,vx,vy,vz]
    df = pd.DataFrame(np.hstack([ts, X]),
                      columns=["timestamp","px","py","pz","vx","vy","vz"])
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] Trajectory -> {out_csv}")

    # Persist collected data (always create files; collector handles empties)
    if 'collector' in locals() and collector is not None:
        coll_dir = args.collect_dir if args.collect_dir else out_dir
        collector.save(coll_dir)

    if nis_rows:
        import csv
        with open(os.path.join(out_dir, "nis_log.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","tracker","nis","S","R_eff"])
            w.writerows(nis_rows)
        # Rolling timeseries for quick visibility
        try:
            df_nis = pd.DataFrame(nis_rows, columns=["timestamp","tracker","nis","S","R_eff"])
            df_nis = df_nis.sort_values("timestamp").reset_index(drop=True)
            df_nis["roll_mean_global"] = df_nis["nis"].rolling(window=200, min_periods=1).mean()
            df_nis["roll_mean_tracker"] = df_nis.groupby("tracker")["nis"].rolling(window=200, min_periods=1).mean().reset_index(level=0, drop=True)
            # Add chi-square(1) reference bands
            df_nis["q50"] = 0.455
            df_nis["q90"] = 2.706
            df_nis["q95"] = 3.841
            df_nis["q99"] = 6.635
            df_nis.to_csv(os.path.join(out_dir, "nis_timeseries.csv"), index=False)
        except Exception:
            pass

    # Save action suggestions CSV
    if action_rows:
        import csv
        with open(os.path.join(out_dir, "action_suggestions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["timestamp","tracker","dx","dy","dz"])
            w.writerows(action_rows)

    if 'w_file' in locals() and w_file is not None:
        try:
            w_file.close()
        except Exception:
            pass

    # Save planner EIG diagnostics if available
    if eig_scores_rows:
        try:
            import csv
            with open(os.path.join(out_dir, "planner_eig_scores.csv"), "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["timestamp","tracker","eig_score","R_eff"])
                w.writerows(eig_scores_rows)
        except Exception:
            pass

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
    p.add_argument("--los_window", type=float, default=0.0,
                   help="If >0, use ±this many seconds around t to compute LOS/IQR (range update still uses exact t)")
    p.add_argument("--use_cir", action="store_true", help="If available, enable CIR for LOS classifier")
    p.add_argument("--los_verbose", action="store_true", help="Print one-time LOS adapter diagnostics")
    p.add_argument("--los_influence", type=float, default=0.2, help="Strength of LOS->reliability (0..1)")
    p.add_argument("--geom_influence", type=float, default=0.4, help="Strength of |e_z|->reliability (0..1)")
    p.add_argument("--ema_alpha", type=float, default=0.0, help="EMA for innovation whiteness (MeasureAdapter)")
    p.add_argument("--r_min_scale", type=float, default=0.5, help="Lower bound on R scaling")
    p.add_argument("--r_max_scale", type=float, default=6.0, help="Upper bound on R scaling")
    p.add_argument("--online_tune", action="store_true", help="Enable OnlineTuner (causal self-calibration)")
    p.add_argument("--online_r_min_scale", type=float, default=0.75)
    p.add_argument("--online_r_max_scale", type=float, default=10.0)
    p.add_argument("--gate_target", type=float, default=0.97)
    p.add_argument("--gate_sigma_init", type=float, default=3.0)
    p.add_argument("--q_adapt", action="store_true", help="Let tuner gently scale process noise (Q)")
    p.add_argument("--init_window", type=int, default=0,
                   help="Use first N timesteps to robustly initialize target position (0=off)")
    p.add_argument("--smooth", action="store_true", help="Enable RTS smoothing after filtering")
    p.add_argument("--smooth_sigma_a_xy", type=float, default=1.0)
    p.add_argument("--smooth_sigma_a_z", type=float, default=0.7)
    p.add_argument("--biasnet_dir", default=None, help="Directory with biasnet.pt and biasnet_meta.json")
    p.add_argument("--bias_gain", type=float, default=0.6, help="Trust in learned bias (0..1)")
    p.add_argument("--fusionnet_dir", default=None, help="Directory with fusionnet.pt and fusionnet_meta.json")
    p.add_argument("--out", default="outputs_swarm")
    p.add_argument("--collect_bias", action="store_true", help="Collect BiasNet samples during run")
    p.add_argument("--collect_fusion", action="store_true", help="Collect FusionNet snaps during run")
    p.add_argument("--collect_dir", default=None, help="Override output dir for collected data (default: experiment out dir)")
    p.add_argument("--decentralized", action="store_true",
                   help="Use decentralized gossip CI instead of centralized CI")
    p.add_argument("--comm_p", type=float, default=1.0, help="Link probability in comms graph")
    p.add_argument("--comm_drop", type=float, default=0.0, help="Packet drop probability per edge per round")
    p.add_argument("--comm_rounds", type=int, default=1, help="Consensus rounds per timestep")
    p.add_argument("--comm_seed", type=int, default=0, help="Seed for comms graph randomness")
    p.add_argument("--budget_k", type=int, default=None,
                   help="If set, only the top-k trackers (by FusionNet weight or reliability) update and fuse")
    p.add_argument("--planner", choices=["heuristic","eig"], default="heuristic")
    p.add_argument("--control_mode", choices=["none","sim","mavsdk"], default="none")
    p.add_argument("--control_rate", type=int, default=5)
    p.add_argument("--r_floor_blend", type=float, default=0.5, help="Blend factor for soft R floor (0..1)")
    args = p.parse_args()
    main(args)
