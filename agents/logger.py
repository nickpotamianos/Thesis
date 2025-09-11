# agent/logger.py
import argparse, time, collections, json, os
import numpy as np
import pandas as pd

# Devkit for GT/eval
from miluv.data import DataLoader
import miluv.utils as utils

from swarm_net.udp import make_rx, make_tx, UdpGroup
from swarm_ml.fusion import CIFuser, CIFuserConfig
from swarm_ml.distrib_ci import GossipFuser, CommsConfig
from swarm_ml.evaluation_swarm import evaluate_and_save
from swarm_ml.features import se_translation_from_matrix
from swarm_ml.planning import suggest_vantage_moves, suggest_vantage_moves_eig

def _load_fusionnet(path_dir: str):
    import os, json, torch
    from swarm_ml.models import FusionNet
    with open(os.path.join(path_dir, "fusionnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    x_mu, x_std = meta.get("x_mu", None), meta.get("x_std", None)
    m = FusionNet(in_dim)
    state = torch.load(os.path.join(path_dir, "fusionnet.pt"), map_location="cpu")
    m.load_state_dict(state)
    m.eval()
    # expose expected input feature dim for assertion later
    try:
        m.input_dim = int(in_dim)
    except Exception:
        pass
    # Install normalizer if available
    try:
        if x_mu is not None and x_std is not None:
            m.set_normalizer(x_mu, x_std)
    except Exception as e:
        print(f"[LOGGER] Warning: could not set FusionNet normalizer: {e}")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--udp", default="239.0.0.1:5001")
    ap.add_argument("--exp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--trackers", default=None, help="Comma-separated tracker ids; if omitted, inferred from data")
    ap.add_argument("--method", choices=["uniform","grid","learned","gossip"], default="grid")
    ap.add_argument("--ci_objective", choices=["logdet","trace"], default="logdet")
    ap.add_argument("--ci_grid", type=float, default=0.1)
    ap.add_argument("--rounds", type=int, default=3, help="gossip rounds if --method=gossip")
    ap.add_argument("--fusionnet_dir", default=None, help="Required if --method learned")
    ap.add_argument("--out", default="outputs_decentralized")
    ap.add_argument("--timeout_ms", type=int, default=300, help="Fuse if not all trackers arrive within this gap")
    ap.add_argument("--fanout", action="store_true", help="Broadcast fused state to nodes")
    ap.add_argument("--planner", choices=["none","heuristic","eig"], default="none",
                    help="If not 'none', compute vantage moves at each fused timestamp and broadcast them")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Load GT sequence for evaluation and define timestamps
    miluv = DataLoader(args.exp, exp_dir="./data/three_robots", cir=False, barometer=False,
                       height=False, imu="px4", cam=None, mag=False)
    data = miluv.data
    robots = list(data.keys())
    if args.target not in robots:
        raise ValueError(f"--target={args.target} not in {robots}")

    # Infer trackers from log if not provided
    if args.trackers:
        trackers = [s.strip() for s in args.trackers.split(",") if s.strip()]
    else:
        trackers = sorted([r for r in robots if r != args.target])

    # Precompute GT target positions at the timestamps of interest
    # Use the union of all UWB timestamps across robots
    uwb_range = pd.concat([data[r]["uwb_range"].assign(robot=r) for r in robots], ignore_index=True)
    query_timestamps = np.sort(uwb_range["timestamp"].unique())
    gt_se23 = utils.get_se23_poses(
        data[args.target]["mocap_quat"](query_timestamps),
        data[args.target]["mocap_pos"].derivative(nu=1)(query_timestamps),
        data[args.target]["mocap_pos"](query_timestamps)
    )
    gt_target_pos = np.array([se_translation_from_matrix(T) for T in gt_se23])

    # CI fusers
    fuser = CIFuser(CIFuserConfig(objective=args.ci_objective, grid_step=args.ci_grid))
    if args.method == "learned":
        fuser.weight_model = _load_fusionnet(args.fusionnet_dir)
        if fuser.weight_model is None:
            raise ValueError("--fusionnet_dir is required for learned method")
    gossip = GossipFuser(CommsConfig(rounds=args.rounds, p_link=1.0, p_drop=0.0, seed=0))

    # UDP receiver
    ip, port = args.udp.split(":")
    rx = make_rx(UdpGroup(mcast_ip=ip, port=int(port)))
    tx_fused = make_tx(UdpGroup(mcast_ip=ip, port=int(port))) if args.fanout else None

    # Accumulators
    window = collections.defaultdict(dict)         # t -> {rid: (mu,P,X,r_eff)}
    last_seen_t = {}                               # rid -> last timestamp
    fused_mu, fused_P, fused_t = [], [], []
    fusion_weights_file = None
    fusion_weights_writer = None
    fusion_cols = None
    # Basic per-tracker counters
    recv_count = collections.Counter()
    missing_at_fuse = collections.Counter()

    # Output directory per (exp,target)
    out_dir = os.path.join(args.out, f"{args.exp}_{args.target}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[LOGGER] listening on {args.udp}; exp={args.exp}; target={args.target}; "
          f"trackers={trackers}; method={args.method}; timeout_ms={args.timeout_ms}; "
          f"planner={args.planner}")

    def maybe_fuse(tb: float):
        # add this line right after the def
        nonlocal fusion_weights_file, fusion_weights_writer, fusion_cols
        # fuse when either (a) all trackers present or (b) timeout since the earliest arrival
        parts = {}
        node_feats = {}
        present = list(window[tb].keys())
        if not present:
            return False
        # TIMEOUT: if we have at least one and the oldest msg at this t waited > timeout_ms
        ready = False
        if all(trk in present for trk in trackers):
            ready = True
        else:
            earliest = min(last_seen_t.get(trk, tb) for trk in present)
            if (time.time() - earliest) * 1000.0 > args.timeout_ms:
                ready = True
        if not ready:
            return False

        # Build parts and node features from what we have
        for rid, payload in window[tb].items():
            mu = np.asarray(payload["mu"], float)
            P  = np.asarray(payload["P"], float)
            X  = np.asarray(payload.get("X", np.zeros(8)), float)
            parts[rid] = (mu, P)
            node_feats[rid] = X
        # Optional: tracker positions and per-node effective R at this tb
        tracker_pos = {rid: np.asarray(payload.get("p", []), float).reshape(-1)
                       for rid, payload in window[tb].items() if "p" in payload}
        r_eff_map   = {rid: float(payload.get("r_eff", 0.35**2))
                       for rid, payload in window[tb].items()}

        # Assert FusionNet feature dimensionality matches model expectation
        if (args.method == "learned") and (fuser.weight_model is not None):
            any_feats = next(iter(node_feats.values()), None)
            if any_feats is not None:
                expected = int(getattr(fuser.weight_model, "input_dim", len(any_feats)))
                assert len(any_feats) == expected, (
                    f"FusionNet expects {expected} features, got {len(any_feats)}. "
                    f"Feature vector (ordered): [var_pos, rel, z_center, R_eff, geom_ez, los_score, gate_sigma, nis_ema] = {any_feats.tolist()}"
                )

        # Fuse (centralized or gossip) and record
        if args.method == "gossip":
            # Use learned weights when fusionnet is provided
            if (fuser.weight_model is not None) and (len(node_feats) > 0):
                keys = list(parts.keys())
                X = np.vstack([node_feats[k].reshape(1, -1) for k in keys])
                w_vec = fuser.weight_model.predict_weights(X)           # (N,)
                w_map = {keys[i]: float(w_vec[i]) for i in range(len(keys))}
                mu_star, P_star, w = gossip.fuse(parts, weights=w_map)
            else:
                mu_star, P_star, w = gossip.fuse(parts)
        else:
            method = args.method if args.method in ("uniform","grid") else "learned"
            mu_star, P_star, w = fuser.fuse(parts, method=method, node_features=node_feats if method=="learned" else None)

        fused_mu.append(mu_star)
        fused_P.append(P_star)
        fused_t.append(float(tb))

        # NEW: broadcast fused state for nodes to use in gating/features
        if tx_fused:
            tx_fused({
                "v": 1,
                "type": "fused",
                "exp": args.exp,
                "target": args.target,
                "t": float(tb),
                "mu": mu_star.tolist(),
                "P":  P_star.tolist()
            })

        # Log weights: stable header order matches declared trackers
        try:
            import csv
            if fusion_weights_file is None:
                fusion_weights_file = open(os.path.join(out_dir, "fusion_weights.csv"), "w", newline="")
                fusion_weights_writer = csv.writer(fusion_weights_file)
                fusion_cols = [f"w_{rid}" for rid in trackers]
                fusion_weights_writer.writerow(["timestamp"] + fusion_cols)
            # write NaN when a tracker's weight is missing at this tb
            row = [float(tb)] + [float(w.get(rid, float('nan'))) for rid in trackers]
            fusion_weights_writer.writerow(row)
        except Exception as e:
            print(f"[LOGGER] fusion weights log error at t={tb:.3f}: {e}")

        # Count missing trackers at fuse time (for diagnostics)
        try:
            for rid in trackers:
                if rid not in present:
                    missing_at_fuse[rid] += 1
        except Exception:
            pass

        # Done with this t
        window.pop(tb, None)

        # --- Optional active sensing plan (requires --planner and at least one tracker position) ---
        try:
            if tx_fused and args.planner != "none" and len(tracker_pos) > 0:
                if args.planner == "eig":
                    moves = suggest_vantage_moves_eig(mu_star, P_star, tracker_pos, r_eff_map)
                else:
                    moves = suggest_vantage_moves(mu_star[:3], tracker_pos)
                # Fanout the command to all nodes
                tx_fused({
                    "v": 1,
                    "type": "cmd_move",
                    "exp": args.exp,
                    "target": args.target,
                    "t": float(tb),
                    "moves": {rid: [float(d) for d in mv] for rid, mv in moves.items()}
                })
        except Exception as e:
            print(f"[LOGGER] planning error at t={tb:.3f}: {e}")
        return True

    # ---- Main receive loop ----
    start_wall = time.time()
    last_progress = start_wall
    try:
        while True:
            m = rx()
            # Schema/version filter
            if int(m.get("v", 0)) != 1:
                continue
            m_type = m.get("type", "local")
            # Ignore our own fanouts and planner commands
            if m_type in ("fused", "cmd_move"):
                continue
            # Only tracker updates are eligible for fusion
            if m_type != "local":
                continue
            # Drop foreign experiment/target traffic on the same multicast group
            if (m.get("exp") != args.exp) or (m.get("target") != args.target):
                continue
            # Required fields present?
            if not all(k in m for k in ("t", "id", "mu", "P")):
                continue
            tb = float(m["t"])
            rid = str(m["id"])
            mu = m["mu"]; P = m["P"]
            X  = m.get("X", [0.0]*8)
            r_eff = float(m.get("r_eff", 0.35**2))
            p     = m.get("p", None)
            window[tb][rid] = {"mu": mu, "P": P, "X": X, "r_eff": r_eff, "p": p}
            last_seen_t[rid] = time.time()
            recv_count[rid] += 1

            # attempt fusion at this timestamp
            maybe_fuse(tb)

            # If we’ve fused all timestamps in the log, we can stop
            if len(fused_t) >= len(query_timestamps):
                break

            # Light heartbeat
            now = time.time()
            if now - last_progress > 1.0:
                print(f"[LOGGER] fused={len(fused_t)} / {len(query_timestamps)}")
                last_progress = now
    except KeyboardInterrupt:
        print("[LOGGER] interrupted; finalizing...")
    finally:
        # Close weights file if opened
        try:
            if fusion_weights_file is not None:
                fusion_weights_file.close()
        except Exception:
            pass

    if not fused_mu:
        print("[LOGGER] no fused states produced.")
        return

    # Align to GT timeline (map fused t to nearest GT index)
    fused_t_arr = np.asarray(fused_t, float)
    mu_arr = np.asarray(fused_mu)
    P_arr  = np.asarray(fused_P)

    # Save trajectory, evaluate, and write summary—same as centralized
    df = pd.DataFrame(np.hstack([fused_t_arr.reshape(-1,1), mu_arr]),
                      columns=["timestamp","px","py","pz","vx","vy","vz"])
    out_csv = os.path.join(out_dir, "target_state.csv")
    df.to_csv(out_csv, index=False)
    print(f"[SAVE] Trajectory -> {out_csv}")

    # Interpolate GT to fused timestamps (nearest)
    idx = np.searchsorted(query_timestamps, fused_t_arr, side="left")
    idx = np.clip(idx, 0, len(gt_target_pos)-1)
    gt_sel = gt_target_pos[idx]

    rm, nees_val = evaluate_and_save(mu_arr, P_arr, gt_sel, out_dir)

    # Minimal roles manifest
    with open(os.path.join(out_dir, "roles.json"), "w") as f:
        json.dump({"target": args.target, "trackers": trackers}, f, indent=2)

    # Diagnostics summary
    try:
        total_fused = len(fused_t)
        print(f"[STATS] fused_timestamps={total_fused} / query={len(query_timestamps)}")
        print(f"[STATS] recv_count={dict(recv_count)}")
        print(f"[STATS] missing_at_fuse={dict(missing_at_fuse)}")
    except Exception:
        pass

    print(f"[DONE] Results written to: {out_dir}")

if __name__ == "__main__":
    main()
