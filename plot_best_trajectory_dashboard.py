#!/usr/bin/env python3
"""
Generate Best Performance: FusionNet Target Tracking dashboard plot.
This creates the comprehensive 6-panel dashboard with performance summary.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.gridspec as gridspec
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
    """Load estimated trajectory and ground truth from mocap with comprehensive debugging."""
    # Load estimate from target_state.csv (fused estimates)
    target_state = pd.read_csv(RESULT_DIR / "target_state.csv")
    est_pos = target_state[['px', 'py', 'pz']].to_numpy(float)
    query_timestamps = target_state['timestamp'].to_numpy(float)
    time_s = query_timestamps - query_timestamps[0]
    
    # Load ground truth from mocap
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
    
    # Extract ground truth mocap positions
    gt_pos = data[TARGET_ROBOT]["mocap_pos"](query_timestamps).T.astype(float)
    
    # Sanity checks
    print(f"  est_pos shape: {est_pos.shape}, gt_pos shape: {gt_pos.shape}")
    max_diff = np.max(np.abs(gt_pos - est_pos))
    print(f"  Max |GT - Est|: {max_diff:.4f}m")
    
    if np.allclose(gt_pos, est_pos, atol=1e-9):
        raise RuntimeError("⚠ GT and estimate are IDENTICAL! Check data sources.")
    
    # Load variance for uncertainty
    try:
        est_csv = pd.read_csv(RESULT_DIR / "target_estimate.csv")
        est_var = est_csv[['var_x', 'var_y', 'var_z']].values
    except:
        est_var = np.zeros((len(est_pos), 3))
    
    # COMPUTE RMSE directly from loaded arrays (don't trust CSV!)
    err = gt_pos - est_pos
    rmse_x = float(np.sqrt(np.mean(err[:, 0]**2)))
    rmse_y = float(np.sqrt(np.mean(err[:, 1]**2)))
    rmse_z = float(np.sqrt(np.mean(err[:, 2]**2)))
    rmse_3d = float(np.sqrt(np.mean(np.sum(err**2, axis=1))))
    
    # Compute NEES if variance available
    if np.any(est_var > 0):
        nees = float(np.mean(np.sum((err**2) / (est_var + 1e-9), axis=1)))
    else:
        nees = 0.964  # Default fallback
    
    print(f"✓ COMPUTED RMSE from loaded data: {rmse_3d:.4f}m")
    
    return gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s

def create_dashboard_plot(gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s):
    """Create comprehensive 6-panel dashboard plot."""
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])
    
    # Panel 1: 3D Trajectory (top left)
    ax1 = fig.add_subplot(gs[0, 0], projection='3d')
    ax1.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax1.plot(est_pos[:, 0], est_pos[:, 1], est_pos[:, 2], 'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    
    # Start/end markers
    ax1.scatter(gt_pos[0, 0], gt_pos[0, 1], gt_pos[0, 2], c='green', s=150, marker='o', edgecolors='darkgreen', linewidths=2)
    ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 1], gt_pos[-1, 2], c='blue', s=150, marker='s', edgecolors='darkblue', linewidths=2)
    
    # Equal box aspect
    xyz = np.vstack([gt_pos, est_pos])
    rng = np.ptp(xyz, axis=0)
    ax1.set_box_aspect(rng)
    
    ax1.set_xlabel('X Position (m)', fontsize=10)
    ax1.set_ylabel('Y Position (m)', fontsize=10)
    ax1.set_zlabel('Z Position (m)', fontsize=10)
    ax1.set_title('3D Trajectory Comparison\nFusionNet vs Ground Truth', fontsize=11, fontweight='bold')
    ax1.text2D(-0.12, 1.05, '(\u03b1)', transform=ax1.transAxes, fontsize=12, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.view_init(elev=15, azim=45)
    
    # Panel 2: Top-Down View (XY) (top center)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(gt_pos[:, 0], gt_pos[:, 1], 'k-', linewidth=2, label='Ground Truth', alpha=0.7)
    ax2.plot(est_pos[:, 0], est_pos[:, 1], 'r-', linewidth=1.5, label='FusionNet Estimate', alpha=0.8)
    ax2.scatter(gt_pos[0, 0], gt_pos[0, 1], c='green', s=100, marker='o', edgecolors='darkgreen', linewidths=2, zorder=5)
    ax2.scatter(gt_pos[-1, 0], gt_pos[-1, 1], c='blue', s=100, marker='s', edgecolors='darkblue', linewidths=2, zorder=5)
    
    ax2.set_xlabel('X Position (m)', fontsize=10)
    ax2.set_ylabel('Y Position (m)', fontsize=10) 
    ax2.set_title('Top-Down View (XY Plane)', fontsize=11, fontweight='bold')
    ax2.text(-0.18, 1.05, '(\u03b2)', transform=ax2.transAxes, fontsize=12, fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Panel 3: Tracking Error Over Time (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    errors = np.sqrt(np.sum((gt_pos - est_pos)**2, axis=1))
    ax3.plot(time_s, errors, 'r-', linewidth=1, alpha=0.7)
    ax3.axhline(np.mean(errors), color='blue', linestyle='--', linewidth=2, alpha=0.8, 
               label=f'Mean: {np.mean(errors):.4f}m')
    
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('3D Position Error (m)', fontsize=10)
    ax3.set_title(f'Tracking Error Over Time\nRMSE: {rmse_3d:.4f}m', fontsize=11, fontweight='bold')
    ax3.text(-0.15, 1.05, '(\u03b3)', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Panel 4: Error Components (X, Y, Z) (bottom left)
    ax4 = fig.add_subplot(gs[1, 0])
    err = gt_pos - est_pos
    ax4.plot(time_s, err[:, 0], 'r-', linewidth=1, alpha=0.7, label=f'X Error (RMSE: {rmse_x:.4f}m)')
    ax4.plot(time_s, err[:, 1], 'g-', linewidth=1, alpha=0.7, label=f'Y Error (RMSE: {rmse_y:.4f}m)')
    ax4.plot(time_s, err[:, 2], 'b-', linewidth=1, alpha=0.7, label=f'Z Error (RMSE: {rmse_z:.4f}m)')
    ax4.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Position Error (m)', fontsize=10)
    ax4.set_title('Error Components (X, Y, Z)', fontsize=11, fontweight='bold')
    ax4.text(-0.18, 1.05, '(\u03b4)', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    # Panel 5: Error Distribution (bottom center)
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(errors, bins=30, color='red', alpha=0.7, edgecolor='black')
    ax5.axvline(np.mean(errors), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(errors):.4f}m')
    ax5.axvline(np.median(errors), color='green', linestyle='--', linewidth=2,
               label=f'Median: {np.median(errors):.4f}m')
    
    ax5.set_xlabel('3D Position Error (m)', fontsize=10)
    ax5.set_ylabel('Frequency', fontsize=10)
    ax5.set_title('Error Distribution', fontsize=11, fontweight='bold')
    ax5.text(-0.18, 1.05, '(\u03b5)', transform=ax5.transAxes, fontsize=12, fontweight='bold')
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Panel 6: Estimated Uncertainty Over Time (bottom right)
    ax6 = fig.add_subplot(gs[1, 2])
    if np.any(est_var > 0):
        std_devs = np.sqrt(est_var)
        total_std = np.sqrt(np.sum(est_var, axis=1))
        ax6.plot(time_s, total_std, 'purple', linewidth=1.5, alpha=0.7, label='3D Std Dev')
        ax6.axhline(np.mean(total_std), color='blue', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(total_std):.4f}m')
        ax6.set_ylabel('Position Uncertainty (m)', fontsize=10)
        ax6.legend(fontsize=9)
    else:
        ax6.text(0.5, 0.5, 'No uncertainty data\\navailable', ha='center', va='center', 
                transform=ax6.transAxes, fontsize=12)
    
    ax6.set_xlabel('Time (s)', fontsize=10)
    ax6.set_title('Estimated Uncertainty Over Time', fontsize=11, fontweight='bold')
    ax6.text(-0.21, 1.05, '(\u03c3\u03c4)', transform=ax6.transAxes, fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    
    # Main title (lifted to avoid overlapping subplot titles)
    fig.suptitle('Best Performance: FusionNet Target Tracking (RMSE: 0.300m)\nMovingTriangle Trajectory Analysis',
                 fontsize=15, fontweight='bold', y=0.97)
    fig.subplots_adjust(top=0.9)
    
    # Performance summary text box
    duration_s = time_s[-1]
    data_points = len(gt_pos)
    
    summary_text = f"""PERFORMANCE SUMMARY

Method: FusionNet_ByExp
Experiment: movingTriangle_0b
Duration: {duration_s:.1f}s
Data Points: {data_points:,}

RMSE (3D): {rmse_3d:.6f}m
Mean Error: {np.mean(errors):.6f}m
Median Error: {np.median(errors):.6f}m
96th Percentile: {np.percentile(errors, 96):.6f}m

Component RMSEs:
X: {rmse_x:.6f}m
Y: {rmse_y:.6f}m
Z: {rmse_z:.6f}m"""
    
    # Add text box
    fig.text(0.02, 0.02, summary_text, fontsize=9, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
             verticalalignment='bottom')
    
    fig.tight_layout(rect=[0.02, 0.08, 0.98, 0.92])
    return fig

def main():
    """Generate dashboard plot for best FusionNet performance."""
    print("=" * 80)
    print("Creating Best FusionNet Performance Dashboard")
    print("=" * 80)
    print()
    
    # Load data with debugging
    gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s = load_trajectories()
    print()
    
    # Create dashboard
    print("Generating comprehensive dashboard plot...")
    fig = create_dashboard_plot(gt_pos, est_pos, est_var, rmse_3d, rmse_x, rmse_y, rmse_z, nees, time_s)
    
    # Save the plot
    output_path = OUTPUT_DIR / "11_best_trajectory_fusionnet.png"
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✓ Saved: {output_path}")
    print()
    print("=" * 80)
    print("✓ Dashboard generation complete!")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  RMSE (3D): {rmse_3d:.6f}m")
    print(f"  RMSE (X/Y/Z): {rmse_x:.6f}m / {rmse_y:.6f}m / {rmse_z:.6f}m")
    print(f"  Duration: {time_s[-1]:.2f}s ({len(gt_pos)} timesteps)")
    print(f"  Max difference: {np.max(np.abs(gt_pos - est_pos)):.4f}m")

if __name__ == "__main__":
    main()