import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from miluv.data import DataLoader


def _concat_with_robot(data: dict, key: str) -> pd.DataFrame:
    dfs = []
    for robot in data.keys():
        if key in data[robot]:
            dfs.append(data[robot][key].assign(robot=robot))
    if not dfs:
        return pd.DataFrame(columns=["timestamp"])
    return pd.concat(dfs, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--result_dir", required=True,
                    help="Directory containing <exp>_<target>/target_estimate.csv (e.g., outputs_swarm_los_on)")
    ap.add_argument("--exp_dir", default="./data/three_robots",
                    help="Path to MILUV three_robots data root")
    ap.add_argument("--out_name", default="traj_comparison.png")
    args = ap.parse_args()

    out_dir = os.path.join(args.result_dir, f"{args.exp}_{args.target}")
    est_csv = os.path.join(out_dir, "target_estimate.csv")
    if not os.path.exists(est_csv):
        raise FileNotFoundError(f"Estimated trajectory CSV not found: {est_csv}")

    est = pd.read_csv(est_csv)
    est_x = est["x"].to_numpy(dtype=float)
    est_y = est["y"].to_numpy(dtype=float)
    est_z = est["z"].to_numpy(dtype=float)
    N = len(est_x)

    # Load data to reconstruct the query timestamps used during tracking
    miluv = DataLoader(
        args.exp,
        exp_dir=args.exp_dir,
        cir=False,
        barometer=False,
        height=True,
        imu="px4",
        cam=None,
        mag=False,
    )
    data = miluv.data

    uwb_range = _concat_with_robot(data, "uwb_range")
    height_df = _concat_with_robot(data, "height")
    query_timestamps = np.sort(np.unique(np.append(
        uwb_range["timestamp"].to_numpy(),
        height_df["timestamp"].to_numpy() if not height_df.empty else np.array([], dtype=float)
    )))

    # Align GT to the same timeline length as estimates (tracking loop ran for len(query_timestamps))
    if len(query_timestamps) != N:
        # Safety: trim to min length
        M = min(len(query_timestamps), N)
        query_timestamps = query_timestamps[:M]
        est_x = est_x[:M]
        est_y = est_y[:M]
        est_z = est_z[:M]

    # Ground-truth target position from mocap splines
    target = args.target
    gt_pos = data[target]["mocap_pos"](query_timestamps)  # shape 3 x T
    gt_x, gt_y, gt_z = gt_pos[0, :], gt_pos[1, :], gt_pos[2, :]

    # Plot XY and Z vs time
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    axs[0].plot(gt_x, gt_y, label="GT", color="C0")
    axs[0].plot(est_x, est_y, label="Estimate", color="C1", alpha=0.9)
    axs[0].set_aspect("equal", adjustable="box")
    axs[0].set_title(f"XY Trajectory: {args.target}")
    axs[0].set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[0].legend()

    axs[1].plot(query_timestamps, gt_z, label="GT z", color="C0")
    axs[1].plot(query_timestamps, est_z, label="Estimate z", color="C1", alpha=0.9)
    axs[1].set_title("Z vs Time")
    axs[1].set_xlabel("time [s]")
    axs[1].set_ylabel("z [m]")
    axs[1].legend()

    fig.suptitle(f"Trajectory Comparison: {args.exp} / {args.target}")
    fig.tight_layout()

    out_path = os.path.join(out_dir, args.out_name)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


if __name__ == "__main__":
    main()

