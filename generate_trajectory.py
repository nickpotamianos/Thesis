#!/usr/bin/env python3
"""
Regenerate FusionNet trajectory for movingTriangle_0b experiment.
This script runs the exact same code that produced the 0.3001m RMSE result,
but captures the estimated trajectory for visualization.
"""

import sys
import subprocess
from pathlib import Path

# Configuration matching the notebook execution
EXP = "default_3_movingTriangle_0b"
TARGET = "ifo003"
FUSIONNET_DIR = "/home/nick/Thesis/runs/20250910_160447_notebook_cv_eval/models/pattern_fold_1_fn/fusionnet_by_exp"
OUTPUT_DIR = "/home/nick/Thesis/trajectory_regeneration"

# Best flags from notebook
COMMON_FLAGS = [
    "--use_height", "--use_height_tf",
    "--uwb_std", "0.8",
    "--pair_corr", "0.3",
    "--sigma_a_xy", "3.0",
    "--sigma_a_z", "1.5",
    "--ci_method", "grid",
    "--ci_objective", "trace",
    "--los_influence", "0",
    "--geom_influence", "0",
    "--ema_alpha", "0.0",
    "--online_tune",
    "--online_r_min_scale", "0.75",
    "--online_r_max_scale", "3.0",
    "--gate_target", "0.90",
    "--gate_sigma_init", "4.0",
    "--q_adapt",
    "--r_floor_blend", "0.5",
]

# FusionNet specific flags
FUSIONNET_FLAGS = [
    "--ci_method", "learned",
    "--fusionnet_dir", FUSIONNET_DIR
]

def main():
    # Build the command
    cmd = [
        sys.executable, "swarm_target_tracking.py",
        "--exp", EXP,
        "--target", TARGET,
        *COMMON_FLAGS,
        *FUSIONNET_FLAGS,
        "--out", OUTPUT_DIR
    ]
    
    print("=" * 80)
    print("Regenerating FusionNet trajectory for movingTriangle_0b experiment")
    print("=" * 80)
    print(f"\nExperiment: {EXP}")
    print(f"Target: {TARGET}")
    print(f"FusionNet Model: {FUSIONNET_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nCommand:")
    print(" ".join(cmd))
    print("\n" + "=" * 80 + "\n")
    
    # Run the tracking
    result = subprocess.run(cmd, cwd="/home/nick/Thesis")
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("✓ Trajectory generation complete!")
        print("=" * 80)
        print(f"\nResults saved to: {OUTPUT_DIR}")
        print("\nKey files:")
        print(f"  - target_estimate.csv: Estimated trajectory")
        print(f"  - target_state.csv: Ground truth trajectory")
        print(f"  - summary.csv: RMSE and other metrics")
        
        # Verify the files exist
        output_path = Path(OUTPUT_DIR)
        if (output_path / "target_estimate.csv").exists():
            print("\n✓ target_estimate.csv found")
        if (output_path / "target_state.csv").exists():
            print("✓ target_state.csv found")
        if (output_path / "summary.csv").exists():
            print("✓ summary.csv found")
    else:
        print(f"\n✗ Trajectory generation failed with return code {result.returncode}")
        sys.exit(result.returncode)

if __name__ == "__main__":
    main()
