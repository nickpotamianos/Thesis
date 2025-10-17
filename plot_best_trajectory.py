#!/usr/bin/env python3
"""
Plot the best FusionNet trajectory (0.3001m RMSE) against ground truth.
This showcases the tracking performance of the ML-enhanced swarm system.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import sys
sys.path.insert(0, '/home/nick/Thesis')
from miluv.data import DataLoader

# Configuration
EXP_NAME = "default_3_movingTriangle_0b"
TARGET_ROBOT = "ifo003"
RESULT_DIR = Path("/home/nick/Thesis/trajectory_regeneration/default_3_movingTriangle_0b_ifo003")
OUTPUT_DIR = Path("/home/nick/Thesis/thesis_plots")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_trajectories():
    """Load estimated trajectory and ground truth from mocap."""
    # Load estimate from swarm_target_tracking.py output (target_state.csv contains the fused estimates)
    target_state = pd.read_csv(RESULT_DIR / "target_state.csv")
    est_pos = target_state[['px', 'py', 'pz']].to_numpy(float)
    query_timestamps = target_state['timestamp'].to_numpy(float)
    time_s = query_timestamps - query_timestamps[0]  # Real time in seconds from start
    
    # Load ground truth from mocap data
    print("Loading mocap ground truth from experiment data...")
    miluv = DataLoader(
        EXP_NAME,
        exp_dir="/home/nick/Thesis/data/three_robots",
        cir=False,
        barometer=False,
        height=True,
        imu="px4",
        cam=None,
        mag=False
    )
    data = miluv.data
    
    # Extract ground truth mocap positions for target robot
    gt_pos = data[TARGET_ROBOT]["mocap_pos"](query_timestamps).T.astype(float).astype(float)  # Transpose to (N, 3)
    
    # Sanity checks with detailed debugging
    assert est_pos.shape == gt_pos.shape, f"Shape mismatch: est={est_pos.shape}, gt={gt_pos.shape}"
    
    print("\n  === DETAILED DEBUG ===")
    print(f"  RESULT_DIR = {RESULT_DIR.resolve()}")
    print(f"  est_pos head:\n{np.round(est_pos[:3], 4)}")
    print(f"  gt_pos  head:\n{np.round(gt_pos[:3], 4)}")
    
    diff = gt_pos - est_pos
    max_diff = np.max(np.abs(diff))
    print(f"\n  max |GT - Est| = {max_diff:.4f}m")
    
    # Show random samples to prove they differ
    rng = np.random.default_rng(0)
    sample_idx = rng.integers(0, len(est_pos), size=5)
    print("\n     idx      GT(x,y,z)           Est(x,y,z)          Diff")
    for i in sample_idx:
        print(f"  {i:6d}  {gt_pos[i]}  {est_pos[i]}  {diff[i]}")
    
    # Hard guard
    if np.allclose(gt_pos, est_pos, atol=1e-9):
        raise RuntimeError("⚠ GT and estimate are IDENTICAL! Check RESULT_DIR and GT loader.")
    print("  ✓ GT and Estimate are different\n")
    
    # Load variance from target_estimate.csv (if available) for uncertainty plots
    try:
        est_csv = pd.read_csv(RESULT_DIR / "target_estimate.csv")
        est_var = est_csv[['var_x', 'var_y', 'var_z']].values
    except:
        # If target_estimate.csv doesn't exist, use zeros
        est_var = np.zeros((len(est_pos), 3))
    
    # Load summary metrics from CSV (but we'll recompute to verify)
    summary = pd.read_csv(RESULT_DIR / "summary.csv")
    saved_rmse_3d = summary['rmse_3d'].values[0]
    saved_nees = summary['nees'].values[0]
    
    print(f"  Saved summary.csv RMSE: {saved_rmse_3d:.4f}m")
    
    # COMPUTE RMSE directly from loaded arrays (don't trust the CSV!)
    err = gt_pos - est_pos
    rmse_x = float(np.sqrt(np.mean(err[:, 0]**2)))
    rmse_y = float(np.sqrt(np.mean(err[:, 1]**2)))
    rmse_z = float(np.sqrt(np.mean(err[:, 2]**2)))
    rmse_3d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    
    # Compute NEES if variance is available
    if np.any(est_var > 0):
        # NEES = mean((err / σ)^2) for each component
        nees = float(np.mean(np.sum((err**2) / (est_var + 1e-9), axis=1)))
    else:
        nees = saved_nees  # Fall back to saved value if no variance
    
    print(f"\n✓ COMPUTED RMSE from loaded data:")
    print(f"  Ground truth (mocap): {len(gt_pos)} points")
    print(f"  Estimate (target_state.csv): {len(est_pos)} points")
    print(f"  RMSE (3D): {rmse_3d:.4f}m")
    print(f"  RMSE (X/Y/Z): {rmse_x:.4f}m / {rmse_y:.4f}m / {rmse_z:.4f}m")
    print(f"  NEES: {nees:.4f}")
    
    # Verify against saved summary
    if abs(rmse_3d - saved_rmse_3d) > 0.01:
        print(f"\n  ⚠ WARNING: Computed RMSE ({rmse_3d:.4f}m) differs from saved ({saved_rmse_3d:.4f}m)!")
        print(f"  ⚠ Using COMPUTED values for plots (saved summary.csv may be stale/wrong)")
    else:
        print(f"  ✓ Computed RMSE matches saved summary.csv")
    
    return gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s

def plot_3d_trajectory(gt_pos, est_pos, rmse_3d):
    """Create 3D trajectory comparison plot."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot ground truth
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2],
            'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Plot estimate
    ax.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2],
            'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    
    # Mark start and end points
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2],
              c='green', s=200, marker='o', label='GT Start', edgecolors='darkgreen', linewidths=2)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2],
              c='blue', s=200, marker='s', label='GT End', edgecolors='darkblue', linewidths=2)
    ax.scatter(est_pos[0, 0], est_pos[0, 1], est_pos[0, 2],
              c='tab:red', s=120, marker='^', label='Est Start', edgecolors='darkred', linewidths=2)
    ax.scatter(est_pos[-1, 0], est_pos[-1, 1], est_pos[-1, 2],
              c='tab:red', s=120, marker='v', label='Est End', edgecolors='darkred', linewidths=2)
    
    # Equal box aspect for faithful geometry
    xyz = np.vstack([gt_pos, est_pos])
    rng = np.ptp(xyz, axis=0)  # (Δx, Δy, Δz)
    ax.set_box_aspect(rng)
    
    ax.set_xlabel('X Position (m)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=12, fontweight='bold')
    ax.set_zlabel('Z Position (m)', fontsize=12, fontweight='bold')
    ax.set_title(f'FusionNet Target Tracking: movingTriangle Experiment\nRMSE = {rmse_3d:.4f}m',
                 fontsize=14, fontweight='bold', pad=20)
    
    ax.legend(fontsize=11, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    return fig

def plot_2d_projections(gt_pos, est_pos, rmse_x, rmse_y, rmse_z, time_s):
    """Create 2D projection plots (XY, XZ, YZ)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # XY projection
    ax = axes[0, 0]
    ax.plot(gt_pos[:, 0], gt_pos[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(est_pos[:, 0], est_pos[:, 1], 'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    ax.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=150, marker='o', 
              label='Start', edgecolors='darkgreen', linewidths=2, zorder=5)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='blue', s=150, marker='s',
              label='End', edgecolors='darkblue', linewidths=2, zorder=5)
    ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'XY Projection\nRMSE_x = {rmse_x:.4f}m, RMSE_y = {rmse_y:.4f}m',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # XZ projection
    ax = axes[0, 1]
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(est_pos[:, 0], est_pos[:, 2], 'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    ax.scatter(gt_pos[0, 0], gt_pos[0, 2], c='green', s=150, marker='o',
              edgecolors='darkgreen', linewidths=2, zorder=5)
    ax.scatter(gt_pos[-1, 0], gt_pos[-1, 2], c='blue', s=150, marker='s',
              edgecolors='darkblue', linewidths=2, zorder=5)
    ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'XZ Projection\nRMSE_x = {rmse_x:.4f}m, RMSE_z = {rmse_z:.4f}m',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # YZ projection
    ax = axes[1, 0]
    ax.plot(gt_pos[:, 1], gt_pos[:, 2], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax.plot(est_pos[:, 1], est_pos[:, 2], 'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    ax.scatter(gt_pos[0, 1], gt_pos[0, 2], c='green', s=150, marker='o',
              edgecolors='darkgreen', linewidths=2, zorder=5)
    ax.scatter(gt_pos[-1, 1], gt_pos[-1, 2], c='blue', s=150, marker='s',
              edgecolors='darkblue', linewidths=2, zorder=5)
    ax.set_xlabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Z Position (m)', fontsize=11, fontweight='bold')
    ax.set_title(f'YZ Projection\nRMSE_y = {rmse_y:.4f}m, RMSE_z = {rmse_z:.4f}m',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Position error over time
    ax = axes[1, 1]
    errors = np.sqrt(np.sum((gt_pos - est_pos)**2, axis=1))
    ax.plot(time_s, errors, 'r-', linewidth=1, alpha=0.7)
    ax.axhline(np.mean(errors), color='blue', linestyle='--', linewidth=2, 
              label=f'Mean Error: {np.mean(errors):.4f}m')
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Position Error (m)', fontsize=11, fontweight='bold')
    ax.set_title('3D Position Error Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('FusionNet Tracking Performance: movingTriangle Experiment',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_error_components(gt_pos, est_pos, rmse_x, rmse_y, rmse_z, time_s):
    """Plot error components separately."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    errors = gt_pos - est_pos
    
    components = ['X', 'Y', 'Z']
    rmses = [rmse_x, rmse_y, rmse_z]
    colors = ['red', 'green', 'blue']
    
    for i, (ax, comp, rmse, color) in enumerate(zip(axes, components, rmses, colors)):
        ax.plot(time_s, errors[:, i], color=color, linewidth=1, alpha=0.7, label=f'{comp} Error')
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
        ax.axhline(rmse, color=color, linestyle='--', linewidth=2, alpha=0.8,
                  label=f'RMSE_{comp.lower()} = {rmse:.4f}m')
        ax.axhline(-rmse, color=color, linestyle='--', linewidth=2, alpha=0.8)
        
        ax.set_ylabel(f'{comp} Error (m)', fontsize=11, fontweight='bold')
        ax.set_title(f'{comp}-Axis Tracking Error', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3)
    
    axes[2].set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    plt.suptitle('FusionNet Error Components: movingTriangle Experiment',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def plot_uncertainty(est_pos, est_var, rmse_3d, time_s):
    """Plot position uncertainty over time."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    std_devs = np.sqrt(est_var)
    
    # Individual variance components
    ax = axes[0, 0]
    ax.plot(time_s, std_devs[:, 0], 'r-', linewidth=1, alpha=0.7, label='σ_x')
    ax.plot(time_s, std_devs[:, 1], 'g-', linewidth=1, alpha=0.7, label='σ_y')
    ax.plot(time_s, std_devs[:, 2], 'b-', linewidth=1, alpha=0.7, label='σ_z')
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Position Std Dev (m)', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Components Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Total 3D uncertainty
    ax = axes[0, 1]
    total_std = np.sqrt(np.sum(est_var, axis=1))
    ax.plot(time_s, total_std, 'purple', linewidth=1.5, alpha=0.7)
    ax.axhline(np.mean(total_std), color='blue', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(total_std):.4f}m')
    ax.axhline(rmse_3d, color='red', linestyle='--', linewidth=2,
              label=f'RMSE: {rmse_3d:.4f}m')
    ax.set_xlabel('Time (s)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Total 3D Std Dev (m)', fontsize=11, fontweight='bold')
    ax.set_title('Total 3D Uncertainty Over Time', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 3D trajectory with uncertainty
    ax = axes[1, 0]
    # Sample every 100th point for clarity
    sample_idx = np.arange(0, len(est_pos), 100)
    ax.plot(est_pos[:, 0], est_pos[:, 1], 'r-', linewidth=1, alpha=0.5, label='Estimate')
    
    # Plot uncertainty ellipses at sampled points (2σ)
    for idx in sample_idx:
        w = 2 * np.sqrt(est_var[idx, 0])  # 2σ_x
        h = 2 * np.sqrt(est_var[idx, 1])  # 2σ_y
        e = Ellipse(xy=(est_pos[idx, 0], est_pos[idx, 1]), width=w, height=h, angle=0,
                   facecolor='red', alpha=0.12, edgecolor='none')
        ax.add_patch(e)
    
    ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
    ax.set_title('XY Trajectory with Uncertainty', fontsize=12, fontweight='bold')
    ax.axis('equal')
    ax.grid(True, alpha=0.3)
    
    # Uncertainty histogram
    ax = axes[1, 1]
    ax.hist(total_std, bins=50, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(total_std), color='blue', linestyle='--', linewidth=2,
              label=f'Mean: {np.mean(total_std):.4f}m')
    ax.axvline(rmse_3d, color='red', linestyle='--', linewidth=2,
              label=f'RMSE: {rmse_3d:.4f}m')
    ax.set_xlabel('3D Std Dev (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('FusionNet Uncertainty Analysis: movingTriangle Experiment',
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    return fig

def main():
    """Generate all trajectory visualization plots."""
    print("=" * 80)
    print("Plotting Best FusionNet Trajectory")
    print("=" * 80)
    print()
    
    # Load data
    gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s = load_trajectories()
    print()
    
    # Generate plots
    print("Generating plots...")
    
    # 1. 3D trajectory
    print("  [1/4] 3D trajectory comparison...")
    fig1 = plot_3d_trajectory(gt_pos, est_pos, rmse_3d)
    fig1.savefig(OUTPUT_DIR / "11_best_trajectory_3d.png", dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("        ✓ Saved: 11_best_trajectory_3d.png")
    
    # 2. 2D projections
    print("  [2/4] 2D projection views...")
    fig2 = plot_2d_projections(gt_pos, est_pos, rmse_x, rmse_y, rmse_z, time_s)
    fig2.savefig(OUTPUT_DIR / "12_best_trajectory_2d.png", dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("        ✓ Saved: 12_best_trajectory_2d.png")
    
    # 3. Error components
    print("  [3/4] Error component analysis...")
    fig3 = plot_error_components(gt_pos, est_pos, rmse_x, rmse_y, rmse_z, time_s)
    fig3.savefig(OUTPUT_DIR / "13_best_trajectory_errors.png", dpi=300, bbox_inches='tight')
    plt.close(fig3)
    print("        ✓ Saved: 13_best_trajectory_errors.png")
    
    # 4. Uncertainty analysis
    print("  [4/4] Uncertainty analysis...")
    fig4 = plot_uncertainty(est_pos, est_var, rmse_3d, time_s)
    fig4.savefig(OUTPUT_DIR / "14_best_trajectory_uncertainty.png", dpi=300, bbox_inches='tight')
    plt.close(fig4)
    print("        ✓ Saved: 14_best_trajectory_uncertainty.png")
    
    print()
    print("=" * 80)
    print("✓ All trajectory plots generated successfully!")
    print("=" * 80)
    print(f"\nPlots saved to: {OUTPUT_DIR}")
    print("\nSummary:")
    print(f"  Experiment: movingTriangle_0b")
    print(f"  Method: FusionNet_ByExp (Pattern-based CV, Fold 1)")
    print(f"  RMSE (3D): {rmse_3d:.4f}m")
    print(f"  RMSE (X/Y/Z): {rmse_x:.4f}m / {rmse_y:.4f}m / {rmse_z:.4f}m")
    print(f"  NEES: {nees:.4f}")
    print(f"  Duration: {time_s[-1]:.2f}s ({len(gt_pos)} timesteps)")

if __name__ == "__main__":
    main()
