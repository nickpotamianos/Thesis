#!/usr/bin/env python3
"""
Best tuned decentralized ML run for default_3_random_0b.

Key improvements vs corrected baseline:
- rounds=8 gossip (stable, similar to 3 with 2 nodes)
- r_floor_blend=0.0 (trust effective R with safe floor= max(R_eff,R_pair))
- gate_target=0.98 (slightly more permissive, still well-calibrated NEES)
- BiasNet gain=0.6 and FusionNet weights (pattern_fold_7)
"""
import os, subprocess, time

def run_bg(cmd, log):
    print("Starting:", " ".join(cmd))
    with open(log, "w") as f:
        return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd="/home/nick/Thesis")

def main():
    subprocess.run(["pkill", "-f", "agents"], capture_output=True)
    time.sleep(2)
    out = "outputs_swarm/ml_0b_tuned"
    subprocess.run(["rm", "-rf", out], capture_output=True)
    print("=== Decentralized ML (tuned) ===")

    logger = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.logger",
        "--exp", "default_3_random_0b", "--target", "ifo003",
        "--method", "gossip", "--rounds", "8",
        "--ci_objective", "trace", "--fanout",
        "--planner", "none",
        "--fusionnet_dir", "runs/20250910_160447_notebook_cv_eval/models/pattern_fold_7_fn/fusionnet_by_exp",
        "--out", out
    ]
    lp = run_bg(logger, "logger_ml_0b_tuned.log"); time.sleep(3)

    common = [
        "--use_height_tf", "--uwb_std", "0.8", "--pair_corr", "0.3",
        "--sigma_a_xy", "3.0", "--sigma_a_z", "1.5",
        "--los_influence", "0", "--geom_influence", "0",
        "--online_tune", "--online_r_min_scale", "0.75", "--online_r_max_scale", "3.0",
        "--gate_target", "0.98", "--gate_sigma_init", "4.0", "--q_adapt",
        "--r_floor_blend", "0.0",
        "--control_mode", "none",
        "--biasnet_dir", "runs/20250910_160447_notebook_cv_eval/models/pattern_fold_7_bn_time_byexp",
        "--bias_gain", "0.6"
    ]
    n1 = ["/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node", "--id", "ifo001", "--target", "ifo003", "--exp", "default_3_random_0b"] + common
    n2 = ["/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node", "--id", "ifo002", "--target", "ifo003", "--exp", "default_3_random_0b"] + common
    p1 = run_bg(n1, "node1_ml_0b_tuned.log"); time.sleep(1)
    p2 = run_bg(n2, "node2_ml_0b_tuned.log")

    try:
        while lp.poll() is None:
            time.sleep(2)
    finally:
        for p in [lp, p1, p2]:
            try:
                if p.poll() is None:
                    p.terminate(); p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        subprocess.run(["pkill", "-f", "agents"], capture_output=True)

    res = os.path.join(out, "default_3_random_0b_ifo003")
    if os.path.exists(res):
        print("\n=== Tuned ML Results ===")
        for f in os.listdir(res):
            p = os.path.join(res, f)
            if os.path.isfile(p):
                print(f"  {f}: {os.path.getsize(p)} bytes")
        s = os.path.join(res, "summary.csv")
        if os.path.exists(s):
            print("\n=== Tuned Summary ===")
            with open(s) as fh: print(fh.read().strip())
    else:
        print("No results directory found")

if __name__ == "__main__":
    main()