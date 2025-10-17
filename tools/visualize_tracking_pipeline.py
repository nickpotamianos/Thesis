#!/usr/bin/env python3
"""Generate a clean, professional pipeline diagram for the thesis.

Shows the chronological flow of the multi-robot tracking system with clear
stage numbering and color-coded components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt  # type: ignore[import]
import matplotlib.patches as mpatches  # type: ignore[import]


def main(output: Path) -> None:
    """Create the pipeline overview figure."""
    
    # Create figure with clean white background
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis('off')
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Define color palette (soft, professional colors)
    colors = {
        'input': '#E3F2FD',      # Light blue
        'preprocess': '#FFF9C4', # Light yellow
        'ml': '#F3E5F5',         # Light purple
        'filter': '#E8F5E9',     # Light green
        'output': '#FCE4EC',     # Light pink
        'network': '#EEEEEE',    # Light gray
        'arrow': '#424242',      # Dark gray
        'special': '#2196F3',    # Blue for special arrows
    }
    
    # Box styling
    box_style = dict(
        boxstyle='round,pad=0.3',
        facecolor='white',
        edgecolor='#424242',
        linewidth=1.5,
    )
    
    # Helper function to draw a box
    def draw_box(x, y, width, height, text, color, number=None):
        """Draw a rounded box with text."""
        rect = mpatches.FancyBboxPatch(
            (x - width/2, y - height/2),
            width,
            height,
            boxstyle='round,pad=0.15',
            facecolor=color,
            edgecolor='#424242',
            linewidth=1.5,
        )
        ax.add_patch(rect)
        
        # Add text
        if number:
            ax.text(x, y + 0.15, f'{number}', fontsize=9, ha='center', va='center', fontweight='bold')
            ax.text(x, y - 0.15, text, fontsize=8, ha='center', va='center')
        else:
            ax.text(x, y, text, fontsize=9, ha='center', va='center', fontweight='bold')
    
    # Helper function to draw arrow
    def draw_arrow(x1, y1, x2, y2, color='#424242', style='->', lw=1.5):
        """Draw an arrow between two points."""
        ax.annotate(
            '',
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(
                arrowstyle=style,
                lw=lw,
                color=color,
            )
        )
    
    # Title
    ax.text(9, 9.5, 'Multi-Robot Target Tracking System', 
            fontsize=16, ha='center', va='top', fontweight='bold')
    
    # ============ TRACKER NODE LANE (TOP) ============
    y_tracker = 7.5
    ax.text(0.5, y_tracker + 0.8, '1. Tracker Node (per robot)', 
            fontsize=12, ha='left', va='center', fontweight='bold')
    ax.plot([0.5, 17.5], [y_tracker + 0.5, y_tracker + 0.5], 'k-', lw=1, alpha=0.3)
    
    # Tracker boxes
    tracker_x = [1.5, 3, 4.5, 6, 7.5, 9.5, 11.5, 13.5]
    tracker_boxes = [
        ('1.1', 'IMU+EKF\nPredict', colors['input']),
        ('1.2', 'UWB Pair\nSelection', colors['preprocess']),
        ('1.3', 'Huber\nAggregate', colors['preprocess']),
        ('1.4', 'LOS/IQR\nScore', colors['preprocess']),
        ('1.5', 'BiasNet\nCorrection', colors['ml']),
        ('1.6', 'Target IF\nUpdate', colors['filter']),
        ('1.7', 'NIS/EMA\nMetrics', colors['filter']),
        ('1.8', 'Broadcast\nUDP', colors['network']),
    ]
    
    for i, (x, (num, text, color)) in enumerate(zip(tracker_x, tracker_boxes)):
        draw_box(x, y_tracker, 1.2, 0.8, text, color, num)
        if i < len(tracker_x) - 1:
            draw_arrow(x + 0.6, y_tracker, tracker_x[i+1] - 0.6, y_tracker, colors['arrow'])
    
    # Loop-back arrow for next timestamp
    draw_arrow(1.5, y_tracker + 0.4, 0.8, y_tracker + 0.4, colors['arrow'], style='-', lw=1)
    draw_arrow(0.8, y_tracker + 0.4, 0.8, y_tracker + 0.6, colors['arrow'], style='-', lw=1)
    draw_arrow(0.8, y_tracker + 0.6, 1.5 - 0.6, y_tracker + 0.6, colors['arrow'], style='->', lw=1)
    ax.text(0.5, y_tracker + 0.7, 'Next\ntimestamp', fontsize=7, ha='right', va='center', style='italic')
    
    # ============ NETWORK LANE (MIDDLE) ============
    y_network = 4.5
    ax.text(0.5, y_network + 0.8, '2. Network Buffer', 
            fontsize=12, ha='left', va='center', fontweight='bold')
    ax.plot([0.5, 17.5], [y_network + 0.5, y_network + 0.5], 'k-', lw=1, alpha=0.3)
    
    # Network buffer box
    draw_box(13.5, y_network, 2.5, 0.8, '2.1\nTimestamp Buffer\n(wait all / timeout)', colors['network'], None)
    
    # Arrow from tracker to network
    draw_arrow(13.5, y_tracker - 0.4, 13.5, y_network + 0.4, colors['special'], style='-|>', lw=2)
    ax.text(13.8, (y_tracker + y_network) / 2, 'UDP\nmulticast', fontsize=8, ha='left', va='center', 
            color=colors['special'], fontweight='bold')
    
    # ============ FUSION LOGGER LANE (BOTTOM) ============
    y_fusion = 1.5
    ax.text(0.5, y_fusion + 0.8, '3. Fusion Logger', 
            fontsize=12, ha='left', va='center', fontweight='bold')
    ax.plot([0.5, 17.5], [y_fusion + 0.5, y_fusion + 0.5], 'k-', lw=1, alpha=0.3)
    
    # Fusion boxes
    fusion_x = [2, 4, 6.5, 9, 11]
    fusion_boxes = [
        ('3.1', 'Top-k\nBudgeting', colors['preprocess']),
        ('3.2', 'FusionNet\nWeights', colors['ml']),
        ('3.3', 'CI Fusion\n(logdet/grid)', colors['filter']),
        ('3.4', 'Evaluate\n(RMSE/NEES)', colors['output']),
        ('3.5', 'Save\nTrajectory', colors['output']),
    ]
    
    for i, (x, (num, text, color)) in enumerate(zip(fusion_x, fusion_boxes)):
        draw_box(x, y_fusion, 1.5, 0.8, text, color, num)
        if i < len(fusion_x) - 1:
            draw_arrow(x + 0.75, y_fusion, fusion_x[i+1] - 0.75, y_fusion, colors['arrow'])
    
    # Arrow from network to fusion
    draw_arrow(13.5, y_network - 0.4, 13.5, y_fusion + 1.5, colors['arrow'], style='-', lw=1.5)
    draw_arrow(13.5, y_fusion + 1.5, 2 - 0.75, y_fusion + 1.5, colors['arrow'], style='-', lw=1.5)
    draw_arrow(2 - 0.75, y_fusion + 1.5, 2 - 0.75, y_fusion + 0.4, colors['arrow'], style='->', lw=1.5)
    
    # Fanout arrow (optional feedback)
    draw_arrow(11 + 0.75, y_fusion, 14.5, y_fusion, colors['special'], style='-', lw=1.5)
    draw_arrow(14.5, y_fusion, 14.5, y_tracker - 0.4, colors['special'], style='-|>', lw=1.5)
    ax.text(14.8, (y_fusion + y_tracker) / 2, 'Fused state\nfanout\n(optional)', 
            fontsize=8, ha='left', va='center', color=colors['special'], fontweight='bold')
    
    # ============ COLOR LEGEND ============
    legend_x = 15
    legend_y = 7
    ax.text(legend_x, legend_y + 0.5, 'Stage Types', fontsize=11, ha='left', va='top', fontweight='bold')
    
    legend_items = [
        ('Data Input', colors['input']),
        ('Preprocessing', colors['preprocess']),
        ('ML Models', colors['ml']),
        ('Filtering', colors['filter']),
        ('Output/Log', colors['output']),
        ('Network', colors['network']),
    ]
    
    for i, (label, color) in enumerate(legend_items):
        y = legend_y - i * 0.4
        rect = mpatches.Rectangle((legend_x, y - 0.15), 0.3, 0.3, facecolor=color, 
                                   edgecolor='#424242', linewidth=1)
        ax.add_patch(rect)
        ax.text(legend_x + 0.5, y, label, fontsize=9, ha='left', va='center')
    
    # ============ METHOD ANNOTATIONS ============
    annotations = [
        (4.5, y_tracker - 0.65, 'Huber M-estimator'),
        (6, y_tracker - 0.65, 'CIR-based LOS'),
        (7.5, y_tracker - 0.65, 'ML bias correction'),
        (9.5, y_tracker - 0.65, 'EMA adaptive tuning'),
        (2, y_fusion - 0.7, 'A-optimal or ML'),
        (4, y_fusion - 0.7, 'Learned weights'),
        (6.5, y_fusion - 0.7, 'Conservative fusion'),
    ]
    
    for x, y, text in annotations:
        ax.text(x, y, text, fontsize=7, ha='center', va='top', style='italic', color='#666666')
    
    # Save figure
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"[OK] Saved pipeline figure to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("thesis_plots/pipeline_overview.png"),
        help="Output path for the PNG figure",
    )
    args = parser.parse_args()
    main(args.output)
