#!/usr/bin/env python3
"""
Complete Decentralized Test with STRICT baseline settings
Supports both BiasNet and FusionNet with gossip learned weights
"""
import subprocess
import time
import os
import sys
from pathlib import Path

# Optional: point this at your trained BiasNet directory (with biasnet.pt and biasnet_meta.json)
# Using the same models as the notebook for default_3_random_0 (fold 6)
BIASNET_DIR = os.environ.get("BIASNET_DIR", "/home/nick/Thesis/runs/20250910_004423_notebook_cv_eval/models/cv_fold_6_bn_time_byexp")
# Optional: point this at your trained FusionNet directory 
FUSIONNET_DIR = os.environ.get("FUSIONNET_DIR", "/home/nick/Thesis/runs/20250910_004423_notebook_cv_eval/models/cv_fold_6_fn/fusionnet_by_exp")

def run_command_background(cmd, logfile):
    """Run command in background and log output"""
    print(f"Starting: {' '.join(cmd)}")
    with open(logfile, 'w') as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, 
                               cwd="/home/nick/Thesis")
    return proc

def main():
    # Ensure we're in the right environment
    os.chdir("/home/nick/Thesis")
    
    # Clean up any existing processes
    subprocess.run(["pkill", "-f", "agents"], capture_output=True)
    time.sleep(2)
    
    # Remove old output
    subprocess.run(["rm", "-rf", "outputs_swarm/decentralized"], capture_output=True)
    
    print("=== Starting Complete Decentralized Test (STRICT) ===")
    
    # --- LOGGER (gossip CI + objective=trace for parity; rounds=3) ---
    logger_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.logger",
        "--exp", "default_3_random_0",
        "--target", "ifo003", 
        "--method", "gossip",
        "--rounds", "3",
        "--ci_objective", "trace",       # STRICT: objective=trace (note: gossip ignores grid search)
        "--fanout",
        "--planner", "none",
        "--out", "outputs_swarm/decentralized"
    ]
    
    # Add FusionNet if available
    if FUSIONNET_DIR:
        logger_cmd.extend(["--fusionnet_dir", FUSIONNET_DIR])
        print(f"Using FusionNet from: {FUSIONNET_DIR}")
    
    logger_proc = run_command_background(logger_cmd, "logger.log")
    time.sleep(3)  # Let logger initialize
    
    # Shared STRICT settings for nodes
    node_common = [
        "--use_height_tf",                # STRICT: use PX4 height (in authors' EKF and z-only TF)
        "--uwb_std", "0.8",               # STRICT
        "--pair_corr", "0.3",             # STRICT
        "--sigma_a_xy", "3.0",            # STRICT
        "--sigma_a_z",  "1.5",            # STRICT
        "--los_influence", "0",           # STRICT: turn off LOS->reliability shaping
        "--geom_influence", "0",          # STRICT: turn off |e_z|->reliability shaping
        "--online_tune",                  # STRICT: enable OnlineTuner (R-scaling/Q-adapt/gating)
        "--online_r_min_scale", "0.75",   # STRICT
        "--online_r_max_scale", "3.0",    # STRICT (tighter than default)
        "--gate_target", "0.90",          # STRICT
        "--gate_sigma_init", "4.0",       # STRICT
        "--q_adapt",                      # STRICT: allow gentle Q inflation/deflation
        "--control_mode", "none"
    ]
    
    # Optionally mount your BiasNet
    if BIASNET_DIR:
        node_common += ["--biasnet_dir", BIASNET_DIR, "--bias_gain", "0.6"]
        print(f"Using BiasNet from: {BIASNET_DIR}")
    
    # --- NODE 1 ---
    node1_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node",
        "--id", "ifo001",
        "--target", "ifo003",
        "--exp", "default_3_random_0",
    ] + node_common
    
    node1_proc = run_command_background(node1_cmd, "node1.log")
    time.sleep(2)
    
    # --- NODE 2 ---
    node2_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node",
        "--id", "ifo002",
        "--target", "ifo003", 
        "--exp", "default_3_random_0",
    ] + node_common
    
    node2_proc = run_command_background(node2_cmd, "node2.log")
    
    print("All processes started. Waiting for completion...")
    
    # Monitor progress
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    while time.time() - start_time < max_wait:
        # Check if processes are still running
        if logger_proc.poll() is not None:
            print("Logger completed!")
            break
            
        time.sleep(10)
        
        # Show progress from logger log
        try:
            with open("logger.log", "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if "fused=" in last_line:
                        print(f"Progress: {last_line}")
        except:
            pass
    
    # Cleanup
    print("Cleaning up processes...")
    for proc in [logger_proc, node1_proc, node2_proc]:
        if proc.poll() is None:
            proc.terminate()
            time.sleep(1)
            if proc.poll() is None:
                proc.kill()
    
    # Check results
    result_dir = Path("outputs_swarm/decentralized/default_3_random_0_ifo003")
    if result_dir.exists():
        files = list(result_dir.glob("*"))
        print(f"\n=== Results Generated ===")
        for f in files:
            print(f"  {f.name}: {f.stat().st_size} bytes")
            
        # Try to show summary if it exists
        summary_file = result_dir / "summary.csv"
        if summary_file.exists() and summary_file.stat().st_size > 0:
            print(f"\n=== Summary Results ===")
            with open(summary_file) as f:
                print(f.read())
        else:
            print("Summary file not generated or empty")
    else:
        print("No results directory found")
    
    print("\n=== Log outputs ===")
    for logfile in ["logger.log", "node1.log", "node2.log"]:
        if os.path.exists(logfile):
            print(f"\n--- {logfile} (last 10 lines) ---")
            with open(logfile) as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())

if __name__ == "__main__":
    main()