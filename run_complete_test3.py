#!/usr/bin/env python3
"""
Complete Test 3: Run decentralized gossip to full completion
"""
import subprocess
import time
import os
import sys
from pathlib import Path

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
    
    print("=== Starting Complete Decentralized Test ===")
    
    # Start logger
    logger_cmd = [
        "python", "-m", "agents.logger",
        "--exp", "default_3_random_0",
        "--target", "ifo003", 
        "--method", "gossip",
        "--rounds", "3",
        "--ci_objective", "logdet",
        "--fanout",
        "--planner", "none",
        "--out", "outputs_swarm/decentralized"
    ]
    
    logger_proc = run_command_background(logger_cmd, "logger.log")
    time.sleep(3)  # Let logger initialize
    
    # Start first robot node
    node1_cmd = [
        "python", "-m", "agents.robot_node",
        "--id", "ifo001",
        "--target", "ifo003",
        "--exp", "default_3_random_0",
        "--use_height_tf",
        "--use_los", 
        "--online_tune",
        "--control_mode", "none"
    ]
    
    node1_proc = run_command_background(node1_cmd, "node1.log")
    time.sleep(2)
    
    # Start second robot node  
    node2_cmd = [
        "python", "-m", "agents.robot_node",
        "--id", "ifo002",
        "--target", "ifo003", 
        "--exp", "default_3_random_0",
        "--use_height_tf",
        "--use_los",
        "--online_tune", 
        "--control_mode", "none"
    ]
    
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