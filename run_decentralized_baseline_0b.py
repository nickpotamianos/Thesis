#!/usr/bin/env python3
"""
run_decentralized_baseline_0b.py
================================
Decentralized baseline test for default_3_random_0b (no BiasNet, no FusionNet) 
with STRICT parameters for comparison against the enhanced ML versions.
"""

import os
import subprocess
import time

def run_command_background(cmd, log_file):
    """Start a command in background and redirect output to log file."""
    print(f"Starting: {' '.join(cmd)}")
    with open(log_file, "w") as f:
        proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, 
                               cwd="/home/nick/Thesis")
    return proc

def main():
    """Run complete decentralized test with baseline (no ML models) for default_3_random_0b."""
    
    # Clean up any existing processes
    subprocess.run(["pkill", "-f", "agents"], capture_output=True)
    time.sleep(2)
    
    # Remove old output
    subprocess.run(["rm", "-rf", "outputs_swarm/baseline_0b"], capture_output=True)
    
    print("=== Starting Decentralized Baseline Test for default_3_random_0b (STRICT, no ML) ===")
    
    # --- LOGGER (gossip CI + objective=trace, no FusionNet) ---
    logger_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.logger",
        "--exp", "default_3_random_0b",
        "--target", "ifo003", 
        "--method", "gossip",
        "--rounds", "3",
        "--ci_objective", "trace",       # STRICT: objective=trace
        "--fanout",
        "--planner", "none",
        "--out", "outputs_swarm/baseline_0b"
    ]
    # Note: No --fusionnet_dir for baseline
    
    print("Starting logger (baseline gossip CI)...")
    logger_proc = run_command_background(logger_cmd, "logger_baseline_0b.log")
    time.sleep(3)
    
    # --- COMMON NODE PARAMETERS (STRICT baseline, no BiasNet) ---
    node_common = [
        "--use_height_tf",                # STRICT: enable height
        "--uwb_std", "0.8",               # STRICT
        "--pair_corr", "0.3",             # STRICT (lower correlation)
        "--sigma_a_xy", "3.0",            # STRICT
        "--sigma_a_z", "1.5",             # STRICT  
        "--los_influence", "0",           # STRICT (disable LOS)
        "--geom_influence", "0",          # STRICT (disable geom)
        "--online_tune",                  # STRICT
        "--online_r_min_scale", "0.75",   # STRICT
        "--online_r_max_scale", "3.0",    # STRICT (tighter than default)
        "--gate_target", "0.90",          # STRICT
        "--gate_sigma_init", "4.0",       # STRICT
        "--q_adapt",                      # STRICT: allow gentle Q inflation/deflation
        "--control_mode", "none"
    ]
    # Note: No --biasnet_dir for baseline
    
    # --- NODE 1 ---
    node1_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node",
        "--id", "ifo001",
        "--target", "ifo003",
        "--exp", "default_3_random_0b",
    ] + node_common
    
    node1_proc = run_command_background(node1_cmd, "node1_baseline_0b.log")
    time.sleep(2)
    
    # --- NODE 2 ---
    node2_cmd = [
        "/home/nick/Thesis/miluv_env/bin/python", "-m", "agents.robot_node",
        "--id", "ifo002",
        "--target", "ifo003", 
        "--exp", "default_3_random_0b",
    ] + node_common
    
    node2_proc = run_command_background(node2_cmd, "node2_baseline_0b.log")
    
    print("All processes started. Waiting for completion...")
    
    # Monitor progress
    max_wait = 300  # 5 minutes max
    start_time = time.time()
    
    try:
        while logger_proc.poll() is None:
            elapsed = time.time() - start_time
            if elapsed > max_wait:
                print(f"Timeout after {max_wait}s, terminating...")
                break
            time.sleep(2)
        
        print("Logger completed!")
        
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        print("Cleaning up processes...")
        for proc in [logger_proc, node1_proc, node2_proc]:
            try:
                if proc.poll() is None:
                    proc.terminate()
                    proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
        
        # Kill any remaining agent processes
        subprocess.run(["pkill", "-f", "agents"], capture_output=True)
    
    # Check results
    results_dir = "outputs_swarm/baseline_0b/default_3_random_0b_ifo003"
    if os.path.exists(results_dir):
        print("\n=== Results Generated ===")
        for file in os.listdir(results_dir):
            file_path = os.path.join(results_dir, file)
            if os.path.isfile(file_path):
                size = os.path.getsize(file_path)
                print(f"  {file}: {size} bytes")
        
        # Show summary results if available
        summary_path = os.path.join(results_dir, "summary.csv")
        if os.path.exists(summary_path):
            print("\n=== Summary Results ===")
            with open(summary_path, "r") as f:
                print(f.read().strip())
    else:
        print("No results directory found")
    
    # Show log excerpts
    print("\n=== Log outputs ===")
    for log_name in ["logger_baseline_0b.log", "node1_baseline_0b.log", "node2_baseline_0b.log"]:
        print(f"\n--- {log_name} (last 10 lines) ---")
        try:
            with open(log_name, "r") as f:
                lines = f.readlines()
                for line in lines[-10:]:
                    print(line.rstrip())
        except FileNotFoundError:
            print("(file not found)")
        except Exception as e:
            print(f"(error reading file: {e})")

if __name__ == "__main__":
    main()