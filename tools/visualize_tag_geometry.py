#!/usr/bin/env python3
"""Create a thesis-ready visualization of UWB tag placements for a robot.

The script reads the real tag offset calibration (config/uwb/tags.yaml) and a
mocap trajectory snapshot, then visualizes how the calibrated tag positions map
into the world frame at a given timestamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Tuple

import matplotlib.pyplot as plt  # type: ignore[import]
import numpy as np
import pandas as pd
import yaml  # type: ignore[import]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", default="ifo001", help="Robot ID (matches entries in tags.yaml)")
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the recorded dataset directory (e.g. data/three_robots/default_3_random3_0b)",
    )
    parser.add_argument(
        "--timestamp",
        type=float,
        default=None,
        help="Optional mocap timestamp to sample (seconds). Defaults to midpoint of the recording.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output file for the PNG figure. Defaults to <dataset>/<robot>_tag_geometry.png",
    )
    return parser.parse_args()


def quaternion_to_matrix(q: Iterable[float]) -> np.ndarray:
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def load_tag_offsets(tags_path: Path, robot: str) -> Dict[str, np.ndarray]:
    if not tags_path.exists():
        raise SystemExit(f"Tag calibration file not found: {tags_path}")

    with tags_path.open() as fh:
        payload = yaml.safe_load(fh)

    if robot not in payload:
        raise SystemExit(f"Robot '{robot}' not found in {tags_path}")

    offsets = {}
    for tag_id, vector_str in payload[robot].items():
        offsets[tag_id] = np.array(json.loads(vector_str), dtype=float)
    return offsets


def load_pose(dataset: Path, robot: str, desired_ts: float | None) -> Tuple[np.ndarray, np.ndarray, float]:
    mocap_path = dataset / robot / "mocap.csv"
    if not mocap_path.exists():
        raise SystemExit(f"Mocap file not found: {mocap_path}")

    df = pd.read_csv(mocap_path)
    if df.empty:
        raise SystemExit(f"Mocap file is empty: {mocap_path}")

    if desired_ts is None:
        idx = len(df) // 2
    else:
        idx = int((df["timestamp"] - desired_ts).abs().idxmin())

    row = df.iloc[idx]
    position = row[["pose.position.x", "pose.position.y", "pose.position.z"]].to_numpy(dtype=float)
    orientation = row[["pose.orientation.x", "pose.orientation.y", "pose.orientation.z", "pose.orientation.w"]].to_numpy(dtype=float)
    timestamp = float(row["timestamp"])
    rotation = quaternion_to_matrix(orientation)
    return position, rotation, timestamp


def sensor_positions(
    robot_position: np.ndarray,
    rotation: np.ndarray,
    offsets: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    return {tag_id: robot_position + rotation @ offset for tag_id, offset in offsets.items()}


def set_equal_3d(ax: plt.Axes) -> None:
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    center = np.mean(limits, axis=1)
    radius = np.max(np.abs(limits - center[:, None]))
    ax.set_xlim3d([center[0] - radius, center[0] + radius])
    ax.set_ylim3d([center[1] - radius, center[1] + radius])
    ax.set_zlim3d([center[2] - radius, center[2] + radius])


def create_plot(
    dataset: Path,
    robot: str,
    robot_position: np.ndarray,
    rotation: np.ndarray,
    tag_offsets: Dict[str, np.ndarray],
    world_positions: Dict[str, np.ndarray],
    timestamp: float,
    output_path: Path,
) -> None:
    fig = plt.figure(figsize=(7.5, 6.0))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(*robot_position, color="#1f77b4", s=80, label="Robot center")

    for idx, (tag_id, pos) in enumerate(sorted(world_positions.items())):
        label = "Tag positions" if idx == 0 else None
        ax.scatter(*pos, s=60, color="#ff7f0e", label=label)
        ax.plot([robot_position[0], pos[0]], [robot_position[1], pos[1]], [robot_position[2], pos[2]], color="#555555", linewidth=1.2)
        local_offset = tag_offsets[tag_id]
        rel_world = np.round(pos - robot_position, 3).tolist()
        local_vals = np.round(local_offset, 3).tolist()
        ax.text(
            pos[0],
            pos[1],
            pos[2] + 0.02,
            (
                f"Tag {tag_id}"
                f"\nworld shift={rel_world}"
                f"\nbody offset={local_vals}"
            ),
            fontsize=8,
            ha="center",
        )

    # Draw body-frame axes (x: forward, y: left, z: up in this dataset).
    axis_length = 0.25
    axes = {
        "x (forward)": (rotation @ np.array([axis_length, 0.0, 0.0]), "#d62728"),
        "y (left)": (rotation @ np.array([0.0, axis_length, 0.0]), "#2ca02c"),
        "z (up)": (rotation @ np.array([0.0, 0.0, axis_length]), "#9467bd"),
    }
    for label, (vec, color) in axes.items():
        ax.quiver(
            robot_position[0],
            robot_position[1],
            robot_position[2],
            vec[0],
            vec[1],
            vec[2],
            color=color,
            linewidth=2,
            arrow_length_ratio=0.1,
        )
        ax.text(
            robot_position[0] + vec[0],
            robot_position[1] + vec[1],
            robot_position[2] + vec[2],
            label,
            color=color,
            fontsize=9,
        )

    ax.set_xlabel("World X (m)")
    ax.set_ylabel("World Y (m)")
    ax.set_zlabel("World Z (m)")
    ax.set_title(
        f"Tag placement in world frame for {robot}\n"
        f"Dataset: {dataset.name} at t={timestamp:.2f} s"
    )
    ax.legend(loc="upper right", frameon=False)
    ax.grid(True, alpha=0.3)

    ax.set_box_aspect((1, 1, 1))
    set_equal_3d(ax)

    subtitle = (
        "Each tag position equals body pose plus the calibrated offset rotated "
        "into the world frame."
    )
    fig.text(0.01, 0.02, subtitle, fontsize=9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    args = parse_args()
    dataset = args.dataset.resolve()
    if not dataset.exists():
        raise SystemExit(f"Dataset directory not found: {dataset}")

    robot = args.robot
    tags_path = Path("config/uwb/tags.yaml")
    tag_offsets = load_tag_offsets(tags_path, robot)

    robot_position, rotation, timestamp = load_pose(dataset, robot, args.timestamp)
    world_positions = sensor_positions(robot_position, rotation, tag_offsets)

    output = args.output
    if output is None:
        output = dataset / f"{robot}_tag_geometry.png"

    create_plot(dataset, robot, robot_position, rotation, tag_offsets, world_positions, timestamp, output)
    print(f"[OK] Saved figure to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
