# agent/robot_node.py
import argparse, time, json, os
import numpy as np
import pandas as pd

# ---- Authors' devkit ----
from miluv.data import DataLoader
import miluv.utils as utils

# EKF models (authors')
try:
    import examples.ekfutils.imu_three_robots_models as model
except Exception:
    import imu_three_robots_models as model  # local fallback
                               (uwb_range["timestamp"] <= t1) &
                               (uwb_range["robot"] == args.id)]
            pair_df_los = select_pair_rows(df_win,
                                           trk_tags=tag_map.get(args.id, []),
                                           tgt_tags=tag_map.get(args.target, []))    import imu_three_robots_models as model  # local fallback

# ---- Ours ----
from swarm_net.udp import make_tx, make_rx_nb, UdpGroup
from swarm_ml.features import se_translation_from_matrix, build_measurement_features
from swarm_ml.tagmap import infer_tag_ids_by_robot, select_pair_rows, robust_range_aggregate, \
                             robust_tracker_sensor_position, robust_target_offset
from swarm_ml.measure_adapter import MeasureAdapter, AdapterConfig
from swarm_ml.target_filter import TargetIF, IFConfig
from swarm_ml.los_adapter import LOSAdapter, LOSConfig
from swarm_ml.online_tuner import OnlineTuner, OnlineAdaptConfig
from swarm_ml.safety import project_to_safe, SafetyLimits
from swarm_control.bridge import ControlBridge

def _concat_with_robot(data: dict, key: str) -> pd.DataFrame:
    dfs = []
    for robot in data.keys():
        if key in data[robot]:
            dfs.append(data[robot][key].assign(robot=robot))
    if not dfs:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(dfs, ignore_index=True)

def _load_biasnet(path_dir: str):
    if path_dir is None:
        return None
    import os, json, torch
    from swarm_ml.models import BiasNet
    with open(os.path.join(path_dir, "biasnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    m = BiasNet(in_dim)
    state = torch.load(os.path.join(path_dir, "biasnet.pt"), map_location="cpu", weights_only=True)
    m.load_state_dict(state)
    # Restore normalizer if present
    try:
        x_mu  = meta.get("x_mu", None)
        x_std = meta.get("x_std", None)
        if x_mu is not None and x_std is not None:
            m.set_normalizer(x_mu, x_std)
    except Exception:
        pass
    m.eval()
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="This robot id (tracker), e.g., ifo002")
    ap.add_argument("--target", required=True, help="Target robot id, e.g., ifo001")
    ap.add_argument("--exp", required=True, help="Experiment name, e.g., default_3_random2_0")
    ap.add_argument("--udp", default="239.0.0.1:5001")
    ap.add_argument("--uwb_std", type=float, default=None)
    ap.add_argument("--uwb_var", type=float, default=0.35**2)
    ap.add_argument("--pair_corr", type=float, default=0.7)
    ap.add_argument("--huber_delta", type=float, default=0.8)
    ap.add_argument("--tags_per_robot", type=int, default=2)
    ap.add_argument("--sigma_a_xy", type=float, default=1.0)
    ap.add_argument("--sigma_a_z", type=float, default=0.5)
    ap.add_argument("--gate_sigma", type=float, default=3.0)
    ap.add_argument("--use_los", action="store_true")
    ap.add_argument("--los_window", type=float, default=0.0,
                    help="If >0, use ±this many seconds around t to compute LOS/IQR (range update still uses exact t)")
    ap.add_argument("--use_cir", action="store_true")
    ap.add_argument("--los_verbose", action="store_true")
    ap.add_argument("--biasnet_dir", default=None)
    ap.add_argument("--bias_gain", type=float, default=0.6, help="Trust in learned bias (0..1)")
    ap.add_argument("--use_height_tf", action="store_true")
    ap.add_argument("--height_std", type=float, default=0.07)
    ap.add_argument("--r_floor_blend", type=float, default=0.5,
                    help="Blend factor for soft R floor with pair variance (0..1)")
    # expose influences for quick A/B
    ap.add_argument("--los_influence", type=float, default=0.5)
    ap.add_argument("--geom_influence", type=float, default=0.5)
    # online tuner (innovation-driven R scale, adaptive gating, optional Q adapt)
    ap.add_argument("--online_tune", action="store_true")
    ap.add_argument("--online_r_min_scale", type=float, default=0.75)
    ap.add_argument("--online_r_max_scale", type=float, default=10.0)
    ap.add_argument("--gate_target", type=float, default=0.97)
    ap.add_argument("--gate_sigma_init", type=float, default=3.0)
    ap.add_argument("--q_adapt", action="store_true")
    ap.add_argument("--realtime", action="store_true", help="Sleep to approximate wall-time replay")
    ap.add_argument("--allow_same_t_fanout", action="store_true", help="If set, allow using fused state at same timestamp (not recommended)")
    ap.add_argument("--warmstart_once", action="store_true", help="Copy first older fused position into local filter state (one-shot)")
    ap.add_argument("--control_mode", choices=["none","sim","mavsdk"], default="none",
                    help="Where to send cmd_move displacements")
    ap.add_argument("--control_rate", type=int, default=5, help="Setpoint rate (Hz) for ControlBridge")
    args = ap.parse_args()

    ip, port = args.udp.split(":")
    tx = make_tx(UdpGroup(mcast_ip=ip, port=int(port)))
    rx_nb = make_rx_nb(UdpGroup(mcast_ip=ip, port=int(port)))  # non-blocking poll
    ctrl = ControlBridge(mode=args.control_mode, rate_hz=args.control_rate)

    # ----------------- Load data -----------------
    miluv = DataLoader(
        args.exp,
        exp_dir="./data/three_robots",
        cir=False,
        barometer=False,
        height=args.use_height_tf,
        imu="px4",
        cam=None,
        mag=False
    )
    data = miluv.data
    robots = list(data.keys())
    if args.id not in robots or args.target not in robots:
        raise ValueError(f"robots in log: {robots}; --id={args.id}, --target={args.target}")

    uwb_range = _concat_with_robot(data, "uwb_range")
    height_df = _concat_with_robot(data, "height") if args.use_height_tf else pd.DataFrame(columns=["timestamp"])

    # Query timestamps = union of UWB + height
    query_timestamps = np.sort(np.unique(np.append(
        uwb_range["timestamp"].to_numpy(),
        height_df["timestamp"].to_numpy() if not height_df.empty else np.array([], dtype=float)
    )))

    # IMU at query times
    imu_at_q = {r: miluv.query_by_timestamps(query_timestamps, robots=r, sensors="imu_px4")[r]
                for r in robots}
    gyro  = {r: imu_at_q[r]["imu_px4"][["timestamp","angular_velocity.x","angular_velocity.y","angular_velocity.z"]].reset_index(drop=True)
             for r in robots}
    accel = {r: imu_at_q[r]["imu_px4"][["timestamp","linear_acceleration.x","linear_acceleration.y","linear_acceleration.z"]].reset_index(drop=True)
             for r in robots}

    # Ground truth SE_2(3)
    gt_se23 = {
        r: utils.get_se23_poses(
            data[r]["mocap_quat"](query_timestamps),
            data[r]["mocap_pos"].derivative(nu=1)(query_timestamps),
            data[r]["mocap_pos"](query_timestamps)
        )
        for r in robots
    }

    # Authors' multi-robot EKF (to get tracker/target poses online from IMU+UWB+height)
    ekf_history = {
        r: {
            "pose": model.common.MatrixStateHistory(state_dim=5, covariance_dim=9),
            "bias": model.common.VectorStateHistory(state_dim=6),
        } for r in robots
    }
    ekf = model.EKF(
        {r: gt_se23[r][0] for r in robots},  # init from GT (same as central script)
        miluv.anchors,
        miluv.tag_moment_arms
    )

    # Tag id map
    tag_map = infer_tag_ids_by_robot(uwb_range, robots, top_n=args.tags_per_robot)

    # Per-link TargetIF for *this* tracker
    tf_cfg = IFConfig(sigma_a_xy=args.sigma_a_xy, sigma_a_z=args.sigma_a_z, gate_N_sigma=args.gate_sigma)
    tf = TargetIF(x0=np.zeros(6), cfg=tf_cfg)

    # Measurement adaptation (BiasNet/LOS/reliability shaping)
    base_var = (args.uwb_std**2) if (args.uwb_std is not None) else args.uwb_var
    mcfg = AdapterConfig(
        base_range_var=base_var,
        los_influence=args.los_influence,
        geom_influence=args.geom_influence,
        bias_model_gain=args.bias_gain,
        # If an OnlineTuner is active, delegate R scaling to it.
        own_rscale=(not args.online_tune)
    )
    meas_ai = MeasureAdapter(mcfg, bias_model=_load_biasnet(args.biasnet_dir))
    los_adapter = LOSAdapter(LOSConfig(use_cir=args.use_cir, verbose=args.los_verbose)) if args.use_los else None

    # Online tuner for R scaling, adaptive gating, optional Q adapt
    tuner = OnlineTuner(OnlineAdaptConfig(
        ema_alpha=0.05,
        r_min_scale=args.online_r_min_scale, r_max_scale=args.online_r_max_scale,
        gate_target_accept=args.gate_target, gate_sigma_init=args.gate_sigma_init,
        q_adapt=args.q_adapt, q_alpha=0.05, q_gain=0.25
    )) if args.online_tune else None

    # Height aligned to query timestamps for optional z-only correction
    height_at_q = {}
    if args.use_height_tf and not height_df.empty:
        for r in robots:
            hdict = miluv.query_by_timestamps(query_timestamps, robots=r, sensors="height")[r]
            height_at_q[r] = hdict["height"]["range"].to_numpy(dtype=float)

    print(f"[AGENT] robot_node for {args.id}; target={args.target}; exp={args.exp}; udp={args.udp}")
    print(f"[AGENT] tag_map: {tag_map}")

    # Pre-group UWB by timestamp for speed
    uwb_by_t = dict(tuple(uwb_range.groupby("timestamp"))) if "timestamp" in uwb_range.columns else {}

    t0 = time.time()
    last_wall = time.time()

    # last fused state (for geometry & gating only)
    last_fused_mu = None
    last_fused_t  = None
    did_warmstart = False
    EPS = 1e-6
    # neighbors' last known positions for safety projection
    neighbor_pos = {}

    # Prepare NIS log (per-node)
    nis_log_path = os.path.join("logs", f"nis_node_{args.id}.csv")
    try:
        if not os.path.exists(nis_log_path):
            with open(nis_log_path, "w") as f:
                f.write("t,nis\n")
    except Exception:
        pass

    # ----------------- Main loop -----------------
    for i, t in enumerate(query_timestamps):
        # drain any fused broadcasts (non-blocking)
        while True:
            m = rx_nb()
            if m is None:
                break
            # schema check
            if int(m.get("v", 0)) != 1:
                continue
            m_type = m.get("type")
            if m_type == "fused":
                # drop foreign experiments/targets
                if m.get("exp") != args.exp or m.get("target") != args.target:
                    continue
                t_f = float(m.get("t", -1.0))
                # accept only if strictly newer than what we have (drop reorders/dups)
                if (last_fused_t is None) or (t_f > last_fused_t + EPS):
                    last_fused_mu = np.asarray(m["mu"], float)
                    last_fused_t  = t_f
            elif m_type == "local":
                # keep neighbor positions for safety projection
                rid_nb = str(m.get("id", ""))
                if rid_nb and rid_nb != args.id:
                    p_nb = m.get("p", None)
                    if p_nb is not None:
                        try:
                            neighbor_pos[rid_nb] = np.asarray(p_nb, float).reshape(3)
                        except Exception:
                            pass
            elif m_type == "cmd_move":
                # apply only if this cmd belongs to our exp/target and includes our id
                if m.get("exp") != args.exp or m.get("target") != args.target:
                    continue
                moves = m.get("moves", {})
                if args.id in moves:
                    mv = np.asarray(moves[args.id], float).reshape(3)
                    # project to safe displacement using our current tracker/body position
                    p_self = se_translation_from_matrix(ekf.pose[args.id]).reshape(3)
                    mv_safe = project_to_safe(p_self, mv, neighbors=neighbor_pos, limits=SafetyLimits())
                    # execute
                    try:
                        ctrl.send_vantage_moves(float(m.get("t", 0.0)), {args.id: mv_safe})
                        print(f"[AGENT {args.id}] exec cmd_move t={m.get('t'):.2f} mv={mv.tolist()} -> safe={mv_safe.tolist()}")
                    except Exception as e:
                        print(f"[AGENT {args.id}] control error: {e}")
        dt = (t - query_timestamps[i-1]) if i > 0 else 0.0

        # Authors' EKF (all robots) to keep tracker/target poses realistic
        u_dict = {
            r: np.array([
                gyro[r].iloc[i]["angular_velocity.x"], gyro[r].iloc[i]["angular_velocity.y"], gyro[r].iloc[i]["angular_velocity.z"],
                accel[r].iloc[i]["linear_acceleration.x"], accel[r].iloc[i]["linear_acceleration.y"], accel[r].iloc[i]["linear_acceleration.z"]
            ]) for r in robots
        }
        ekf.predict(u_dict, dt)

        df_t = uwb_by_t.get(t, None)
        if df_t is not None and not df_t.empty:
            # Feed all UWB to the authors' EKF (keeps poses tight)
            for _, row in df_t.iterrows():
                ekf.correct({"range": float(row["range"]),
                             "to_id": int(row["to_id"]),
                             "from_id": int(row["from_id"])})
        if args.use_height_tf and (t in height_df["timestamp"].values):
            hrows = height_df[height_df["timestamp"] == t]
            for _, row in hrows.iterrows():
                ekf.correct({"height": float(row["range"]), "robot": str(row["robot"])})

        # Store EKF state history (not strictly required here)
        for r in robots:
            ekf_history[r]["pose"].add(t, ekf.pose[r], ekf.pose_covariance[r])
            ekf_history[r]["bias"].add(t, ekf.bias[r], ekf.bias_covariance[r])

        # This node’s local measurement set: only rows originated by this robot
        pair_df = pd.DataFrame(columns=["timestamp"])  # default empty
        if df_t is not None and not df_t.empty:
            df_r = df_t[df_t["robot"] == args.id]
            if not df_r.empty:
                pair_df = select_pair_rows(df_r, trk_tags=tag_map.get(args.id, []),
                                           tgt_tags=tag_map.get(args.target, []))

        # Local TargetIF update for this tracker
        # 1) predict (with optional Q adaptation from tuner)
        if tuner is not None and args.q_adapt:
            qxy_scale, qz_scale = tuner.get_q_scales()
            tf.cfg.sigma_a_xy = args.sigma_a_xy * float(qxy_scale)
            tf.cfg.sigma_a_z  = args.sigma_a_z  * float(qz_scale)
        tf.predict(float(dt))

        # 2) if we have a measurement, aggregate + adapt + correct
        z_agg_center, R_eff, rel, los_score = 0.0, args.uwb_var, 0.0, 0.5
        tracker_pos = se_translation_from_matrix(ekf.pose[args.id])
        sensor_pos = tracker_pos  # default; overwritten when pair measurement available
        target_pred_pos = None
        if isinstance(last_fused_mu, np.ndarray) and (last_fused_t is not None):
            # causal fanout (default): use only strictly older fused state
            if args.allow_same_t_fanout or (last_fused_t + EPS < float(t)):
                target_pred_pos = last_fused_mu[:3]
                # optional warm-start once (only from strictly older fused)
                if args.warmstart_once and (last_fused_t + EPS < float(t)) and (not did_warmstart):
                    tf.mu[:3] = last_fused_mu[:3]
                    did_warmstart = True
        if pair_df is not None and not pair_df.empty:
            z_agg, R_pair, _ = robust_range_aggregate(
                pair_df, base_var=base_var, rho=args.pair_corr, huber_delta=args.huber_delta
            )
            # Effective sensor position (tracker tag position minus target tag offset)
            sensor_pos = robust_tracker_sensor_position(
                pair_df=pair_df,
                trk=args.id, trk_tags=tag_map.get(args.id, []),
                T_trk=ekf.pose[args.id], tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )
            tgt_offset_w = robust_target_offset(
                pair_df=pair_df, tgt_tags=tag_map.get(args.target, []),
                T_tgt=ekf.pose[args.target], tag_moment_arms=miluv.tag_moment_arms,
                huber_delta=args.huber_delta
            )
            eff_sensor_pos = sensor_pos - tgt_offset_w
            sensor_pos = eff_sensor_pos

            # LOS score first (so features match training)
            if los_adapter is not None:
                try:
                    # use the windowed set so m≥3 is common
                    s = los_adapter.score(pair_df_los, extras=None)
                    if s is not None:
                        los_score = float(s)
                except Exception:
                    pass

            # Optional heights for feature parity (will be ignored if None)
            h_trk = None
            h_tgt = None
            if args.use_height_tf and height_at_q:
                try:
                    if args.id in height_at_q:
                        h_trk = float(height_at_q[args.id][i])
                    if args.target in height_at_q:
                        h_tgt = float(height_at_q[args.target][i])
                except Exception:
                    h_trk = None; h_tgt = None

            # features (use fused target prediction if available)
            # BiasNet features: LOS neutral unless --use_los; include Δz when --use_height_tf
            h_trk_feat = None; h_tgt_feat = None
            if args.use_height_tf and height_at_q:
                try:
                    h_trk_feat = float(height_at_q[args.id][i])
                    h_tgt_feat = float(height_at_q[args.target][i])
                except (KeyError, IndexError, TypeError, ValueError):
                    h_trk_feat = None; h_tgt_feat = None

            feat = build_measurement_features(
                tracker_pos=eff_sensor_pos,
                target_pred_pos=None,          # no bearing/dist inputs
                uwb_range=float(z_agg),
                los_score=los_score if args.use_los else None,
                height_tracker=h_trk_feat,
                height_target=h_tgt_feat
            )
            
            # --- Extend with pair-quality & identity (must match collector) ---
            # Need to import robust_range_aggregate for pair analysis
            from swarm_ml.tagmap import robust_range_aggregate
            _, R_pair_agent, meta_pairs_agent = robust_range_aggregate(
                pair_df, base_var=base_var, rho=args.pair_corr, huber_delta=args.huber_delta
            )
            m_eff = float(meta_pairs_agent.get("m_eff", 1.0))
            # IQR from windowed or current pair data
            if pair_df_los is not None and not pair_df_los.empty:
                zs_los = pair_df_los["range"].to_numpy(dtype=float)
                if zs_los.size >= 3:
                    q25, q75 = np.percentile(zs_los, [25, 75]); iqr = float(max(0.0, q75 - q25))
                elif zs_los.size == 2:
                    iqr = float(abs(zs_los[1] - zs_los[0]))
                else:
                    iqr = float(meta_pairs_agent.get("iqr", 0.0))
            else:
                iqr = float(meta_pairs_agent.get("iqr", 0.0))
            # Create tracker vocabulary and one-hot encoding
            tracker_vocab = sorted([args.id])  # In robot_node, only self is tracker
            if args.target in robots:  # Handle case where target could be tracker too
                all_trackers = [r for r in robots if r != args.target]
                tracker_vocab = sorted(all_trackers)
            onehot = np.zeros(len(tracker_vocab), dtype=float)
            if args.id in tracker_vocab:
                onehot[tracker_vocab.index(args.id)] = 1.0
            feat = np.hstack([feat, [float(R_pair_agent), m_eff, iqr], onehot])

            # tuner pre-setup (R scale & gate)
            link = (args.id, args.target)
            r_scale = 1.0
            if tuner is not None:
                r_scale = tuner.get_r_scale(link)
                meas_ai._rscale[link] = float(r_scale)
                tf.cfg.gate_N_sigma = float(tuner.get_gate_sigma(link))

            # measurement adaptation
            z_corr, R_eff, meta = meas_ai.correct(
                tracker_id=args.id, target_id=args.target, z=float(z_agg),
                tracker_pos=eff_sensor_pos, target_pred_pos=target_pred_pos,
                los_score=los_score, features=feat
            )
            # Soft floor with pair variance (configurable blend)
            k = float(args.r_floor_blend)
            R_eff = k * float(R_eff) + (1.0 - k) * float(max(R_eff, R_pair))
            z_agg_center = float(z_corr)
            rel = float(meta["reliability"])

            # pre‑gate using predicted S with current R_eff
            accepted = True
            if tuner is not None:
                h0, H = tf._range_linearize(tf.mu, eff_sensor_pos)
                from numpy.linalg import inv
                P_pred_local = inv(tf.J)
                S_pred = float(H @ P_pred_local @ H.T + R_eff)
                nu_pred = float(z_corr - h0)
                gate_sigma = float(tuner.get_gate_sigma(link))
                accepted = float((nu_pred * nu_pred) / S_pred) <= gate_sigma * gate_sigma
                tuner.after_gating(link, accepted=accepted)

            upd = tf.correct(z_agg_center, float(R_eff), tracker_pos=eff_sensor_pos) if accepted else {"used": False}
            if upd.get("used", False):
                meas_ai.update_from_innov(args.id, args.target, innov=upd.get("innov", None), S=upd.get("S", None))
                if tuner is not None:
                    tuner.after_update(
                        link=link,
                        innovation=float(upd.get("innov", 0.0)),
                        S_scalar=float(upd.get("S", 1.0)),
                        r_is_maxed=bool(float(r_scale) >= 0.6 * float(tuner.cfg.r_max_scale)),
                        geom_ez=None
                    )
                # Log NIS to CSV
                try:
                    if upd.get("S", None) is not None and upd.get("innov", None) is not None:
                        nis = float((upd["innov"] * upd["innov"]) / upd["S"]) if upd["S"] > 0 else float('nan')
                        with open(nis_log_path, "a") as f:
                            f.write(f"{float(t)},{nis}\n")
                except Exception:
                    pass

            # Optional: z-only height difference
            if args.use_height_tf and height_at_q:
                if args.target in height_at_q and args.id in height_at_q:
                    h_tgt = float(height_at_q[args.target][i])
                    h_trk = float(height_at_q[args.id][i])
                    if np.isfinite(h_tgt) and np.isfinite(h_trk):
                        dz_meas = h_tgt - h_trk
                        R_h = 2.0 * (args.height_std ** 2)
                        z_trk = float(se_translation_from_matrix(ekf.pose[args.id])[2])
                        tf.correct_height(dz_meas, z_trk, R_h)

        mu_i, P_i = tf.posterior()
        # Numerical hygiene: symmetrize and floor tiny negative eigs
        P_i = 0.5 * (P_i + P_i.T)
        me = float(np.linalg.eigvalsh(P_i).min())
        if me < 1e-10:
            P_i = P_i + np.eye(P_i.shape[0]) * (1e-10 - me + 1e-12)

        # Node features for FusionNet (8‑vector)
        var_pos = float(np.trace(P_i[:3, :3]))
        geom_ez = 0.0
        if target_pred_pos is not None:
            diff = (target_pred_pos - sensor_pos)
            nrm = float(np.linalg.norm(diff) + 1e-9)
            geom_ez = float(abs(diff[2]) / nrm)
        gate_sig = float(tf.cfg.gate_N_sigma)
        nis_ema = float(meas_ai._whiten_ema.get((args.id, args.target), 1.0))
        X = np.array([var_pos, rel, z_agg_center, float(R_eff), geom_ez, float(los_score),
                      gate_sig, nis_ema], dtype=float)

        # Broadcast compact message (+ our current effective sensor position 'p')
        tx({
            "v": 1,
            "type": "local",
            "t": float(t),
            "id": str(args.id),
            "target": str(args.target),
            "exp": str(args.exp),
            "mu": mu_i.tolist(),
            "P": P_i.tolist(),
            "X": X.tolist(),
            "r_eff": float(R_eff),
            "p": sensor_pos.tolist()
        })

        # pace if requested
        if args.realtime:
            # roughly follow log rate (assumes ~200Hz timestamps, sleep lightly)
            now = time.time()
            delay = max(0.0, (query_timestamps[min(i+1, len(query_timestamps)-1)] - t))
            # cap to something reasonable to avoid long sleeps when log has gaps
            delay = min(0.05, delay)
            if now - last_wall < delay:
                time.sleep(delay - (now - last_wall))
            last_wall = time.time()

    print(f"[AGENT] {args.id} finished streaming {len(query_timestamps)} timesteps.")

if __name__ == "__main__":
    main()
