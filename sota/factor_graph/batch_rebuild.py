#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, numpy as np, pandas as pd, gtsam
from pathlib import Path
from miluv.data import DataLoader
from sota.uwb.infer_uwb import UwbML
from sota.factor_graph.range_factory_std import RangeFactoryStd

POSE_KEY = lambda r, k: gtsam.symbol(chr(ord('P') + r), k)   # P,Q,R
VEL_KEY  = lambda r, k: gtsam.symbol(chr(ord('V') + r), k)   # V,W,X
BIAS_KEY = lambda r, k: gtsam.symbol(chr(ord('B') + r), k)   # B,C,D

def frd_to_flu(v): return np.array([v[0], -v[1], -v[2]], dtype=float)

def fuse_range_and_cir(robot_table, max_dt=0.05):
    rng = robot_table["uwb_range"][["timestamp","from_id","to_id","range","gt_range"]].copy()
    cir = robot_table["uwb_cir"  ][["timestamp","from_id","to_id","cir"]].copy()
    rng = rng.sort_values("timestamp"); cir = cir.sort_values("timestamp")
    fused = pd.merge_asof(rng, cir, on="timestamp", by=["from_id","to_id"],
                          tolerance=max_dt, direction="nearest").dropna(subset=["cir"])
    return fused

def estimate_initial_rot(imu_df, seconds=0.5):
    imu_df = imu_df.sort_values("timestamp")
    t0 = float(imu_df["timestamp"].iloc[0])
    w = imu_df[imu_df["timestamp"] <= t0 + seconds]
    a = frd_to_flu(np.array([w["linear_acceleration.x"].mean(),
                             w["linear_acceleration.y"].mean(),
                             w["linear_acceleration.z"].mean()]))
    roll  = np.arctan2(a[1], a[2])
    pitch = np.arctan2(-a[0], np.sqrt(a[1]**2 + a[2]**2))
    return gtsam.Rot3.Ypr(0.0, pitch, roll)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", help="experiment name / directory")
    ap.add_argument("--out", default="out/batch_result.values")
    args = ap.parse_args()

    mv = DataLoader(args.exp, exp_dir="./data/three_robots", cir=True, height=False)

    # --- build graph & initial ----------------------------------------
    graph  = gtsam.NonlinearFactorGraph()
    values = gtsam.Values()

    imu_params = gtsam.PreintegrationParams.MakeSharedU(9.81)
    imu_params.setGyroscopeCovariance(np.eye(3)*(0.015**2))
    imu_params.setAccelerometerCovariance(np.eye(3)*(0.019**2))
    imu_params.setIntegrationCovariance(np.eye(3)*(0.0001**2))

    preint = {r: gtsam.PreintegratedImuMeasurements(imu_params, gtsam.imuBias.ConstantBias()) for r in range(3)}
    last_t = {r: None for r in range(3)}
    last_k = {r: 0 for r in range(3)}

    # anchors
    rng_factory = RangeFactoryStd(UwbML(args.exp))
    if mv.anchors:
        anchors_xyz = {aid: np.asarray(pos, float).reshape(3) for aid, pos in mv.anchors.items()}
        rng_factory.load_anchors(graph, values, anchors_xyz=anchors_xyz)

    # priors
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-3]*6))
    for ridx, (name, rob) in enumerate(mv.data.items()):
        k = POSE_KEY(ridx, 0)
        R0 = estimate_initial_rot(rob["imu_px4"])
        P0 = gtsam.Pose3(R0, gtsam.Point3(0,0,0))
        values.insert(k, P0)
        graph.add(gtsam.PriorFactorPose3(k, P0, prior_noise))

        values.insert(VEL_KEY(ridx,0), np.zeros(3))
        values.insert(BIAS_KEY(ridx,0), gtsam.imuBias.ConstantBias())
        v_noise = gtsam.noiseModel.Isotropic.Sigma(3, 1e-3)
        b_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        graph.add(gtsam.PriorFactorVector(VEL_KEY(ridx,0), np.zeros(3), v_noise))
        graph.add(gtsam.PriorFactorConstantBias(BIAS_KEY(ridx,0),
                                                gtsam.imuBias.ConstantBias(), b_noise))

    # timeline
    streams = []
    for ridx, (_, rob) in enumerate(mv.data.items()):
        imu = rob["imu_px4"][["timestamp",
                              "angular_velocity.x","angular_velocity.y","angular_velocity.z",
                              "linear_acceleration.x","linear_acceleration.y","linear_acceleration.z"]].copy()
        imu["sensor"]="imu"; imu["robot"]=ridx
        uwb = fuse_range_and_cir(rob); uwb["sensor"]="uwb"; uwb["robot"]=ridx
        streams.append(imu); streams.append(uwb)
    tl = sorted((r for df in streams for r in df.to_dict("records")), key=lambda d: d["timestamp"])

    # build full graph
    for pkt in tl:
        r = pkt["robot"]
        if pkt["sensor"] == "imu":
            if last_t[r] is not None:
                dt = pkt["timestamp"] - last_t[r]
                acc = frd_to_flu(np.array([pkt["linear_acceleration.x"],
                                           pkt["linear_acceleration.y"],
                                           pkt["linear_acceleration.z"]], dtype=float))
                gyr = frd_to_flu(np.array([pkt["angular_velocity.x"],
                                           pkt["angular_velocity.y"],
                                           pkt["angular_velocity.z"]], dtype=float))
                preint[r].integrateMeasurement(acc, gyr, dt)
            last_t[r] = pkt["timestamp"]

            if preint[r].deltaTij() >= 2.0:
                k_prev = last_k[r]; k_new = k_prev + 1
                graph.add(gtsam.ImuFactor(
                    POSE_KEY(r,k_prev), VEL_KEY(r,k_prev),
                    POSE_KEY(r,k_new),  VEL_KEY(r,k_new),
                    BIAS_KEY(r,k_prev), preint[r]))
                bias_covar = gtsam.noiseModel.Isotropic.Sigma(6, 0.0001)
                graph.add(gtsam.BetweenFactorConstantBias(BIAS_KEY(r,k_prev),
                                                          BIAS_KEY(r,k_new),
                                                          gtsam.imuBias.ConstantBias(), bias_covar))
                # init new states from previous
                values.insert(POSE_KEY(r,k_new), values.atPose3(POSE_KEY(r,k_prev)))
                values.insert(VEL_KEY(r,k_new),  np.zeros(3))
                values.insert(BIAS_KEY(r,k_new), gtsam.imuBias.ConstantBias())
                preint[r].resetIntegrationAndSetBias(gtsam.imuBias.ConstantBias())
                last_k[r] = k_new

        else:
            # uwb
            key_from = POSE_KEY(r, last_k[r])
            to_id = int(pkt["to_id"])
            cir   = pkt["cir"]; raw_range = float(pkt["range"])
            if to_id <= 5:
                rng_factory.add_factor(graph, key_i=key_from, to_id=to_id,
                                       cir_blob=cir, raw_range=raw_range)
            else:
                r_to = [0,1,2][[10,20,30].index(to_id//10*10)]
                key_to = POSE_KEY(r_to, last_k[r_to])
                if key_to != key_from:
                    rng_factory.add_factor(graph, key_i=key_from, to_id=key_to,
                                           cir_blob=cir, raw_range=raw_range)

    print(f"Full graph: {graph.size()} factors, {values.size()} variables")
    prm = gtsam.LevenbergMarquardtParams()
    prm.setVerbosityLM("SUMMARY")
    result = gtsam.LevenbergMarquardtOptimizer(graph, values, prm).optimizeSafely()

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "save"):
        result.save(str(out))
        print("Batch result written to", out.resolve())
    else:
        # text fallback compatible with export_values.py
        from sota.factor_graph.utils_io import _values_save_txt
        _values_save_txt(result, out.with_suffix(".txt"))
        print("Batch result written to", out.with_suffix(".txt").resolve())

if __name__ == "__main__":
    main()
