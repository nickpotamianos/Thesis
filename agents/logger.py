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

def _load_fusionnet(path_dir: str):
    if path_dir is None:
        return None
    import os, json, torch
    from swarm_ml.models import FusionNet
    with open(os.path.join(path_dir, "fusionnet_meta.json"), "r") as f:
        meta = json.load(f)
    in_dim = int(meta["in_dim"])
    m = FusionNet(in_dim)
    state = torch.load(os.path.join(path_dir, "fusionnet.pt"), map_location="cpu")
    m.load_state_dict(state)
    m.eval()
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

    # Output directory per (exp,target)
    out_dir = os.path.join(args.out, f"{args.exp}_{args.target}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[LOGGER] listening on {args.udp}; exp={args.exp}; target={args.target}; trackers={trackers}; method={args.method}")

    def maybe_fuse(tb: float):
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

        # Fuse (centralized or gossip) and record
        if args.method == "gossip":
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
                "t": float(tb),
                "mu": mu_star.tolist(),
                "P":  P_star.tolist()
            })

        # Log weights
        try:
            import csv
            if fusion_weights_file is None:
                fusion_weights_file = open(os.path.join(out_dir, "fusion_weights.csv"), "w", newline="")
                fusion_weights_writer = csv.writer(fusion_weights_file)
                fusion_weights_writer.writerow(["timestamp"] + [f"w_{rid}" for rid in w.keys()])
            else:
                fusion_weights_writer = fusion_weights_writer  # noqa
            fusion_weights_writer.writerow([float(tb)] + [float(w[r]) for r in w.keys()])
        except Exception:
            pass

        # Done with this t
        window.pop(tb, None)
        return True

    # ---- Main receive loop ----
    start_wall = time.time()
    last_progress = start_wall
    try:
        while True:
            m = rx()
            # Ignore our own fused-state broadcasts
            if m.get("type") == "fused":
                continue
            tb = float(m["t"])
            rid = str(m["id"])
            mu = m["mu"]; P = m["P"]
            X  = m.get("X", [0.0]*8)
            r_eff = float(m.get("r_eff", 0.35**2))
            window[tb][rid] = {"mu": mu, "P": P, "X": X, "r_eff": r_eff}
            last_seen_t[rid] = time.time()

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

    print(f"[DONE] Results written to: {out_dir}")

if __name__ == "__main__":
    main()
