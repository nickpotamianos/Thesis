# sota/factor_graph/utils_io.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os, csv, re, pathlib, gtsam
import numpy as np

def ensure_dir(path: str | os.PathLike) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path

# --------- fallback tiny text serializer for Values (poses + vectors) -------
def _values_save_txt(vals: gtsam.Values, path: pathlib.Path) -> None:
    with path.open("w") as f:
        for k in vals.keys():
            # Pose3
            try:
                p = vals.atPose3(k)
                t = p.translation(); R = p.rotation().toQuaternion()
                # Handle both Point3 and numpy array cases
                if hasattr(t, 'x'):
                    tx, ty, tz = t.x(), t.y(), t.z()
                else:
                    tx, ty, tz = float(t[0]), float(t[1]), float(t[2])
                if hasattr(R, 'x'):
                    rx, ry, rz, rw = R.x(), R.y(), R.z(), R.w()
                else:
                    rx, ry, rz, rw = float(R[0]), float(R[1]), float(R[2]), float(R[3])
                f.write(f"{k} POSE3 {tx} {ty} {tz} {rx} {ry} {rz} {rw}\n")
                continue
            except RuntimeError:
                pass
            # Vector3 (velocities)
            try:
                v = vals.atVector(k)
                arr = np.asarray(v, dtype=float).reshape(-1)
                if arr.size == 3:
                    f.write(f"{k} VEC3 {arr[0]} {arr[1]} {arr[2]}\n")
                    continue
            except RuntimeError:
                pass
            # Point3 (anchors)
            try:
                a = vals.atPoint3(k)
                if hasattr(a, 'x'):
                    ax, ay, az = a.x(), a.y(), a.z()
                else:
                    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
                f.write(f"{k} POINT3 {ax} {ay} {az}\n")
                continue
            except RuntimeError:
                pass
            # imuBias.ConstantBias (two vec3)
            try:
                b = vals.atConstantBias(k)
                a = b.accelerometer(); g = b.gyroscope()
                if hasattr(a, 'x'):
                    ax, ay, az = a.x(), a.y(), a.z()
                    gx, gy, gz = g.x(), g.y(), g.z()
                else:
                    ax, ay, az = float(a[0]), float(a[1]), float(a[2])
                    gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
                f.write(f"{k} BIAS {ax} {ay} {az} {gx} {gy} {gz}\n")
                continue
            except (RuntimeError, AttributeError):
                pass
    # done

def _values_load_txt(path: pathlib.Path) -> gtsam.Values:
    vals = gtsam.Values()
    with path.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            k = int(parts[0]); tag = parts[1]
            if tag == "POSE3":
                x,y,z,qx,qy,qz,qw = map(float, parts[2:9])
                pose = gtsam.Pose3(gtsam.Rot3.Quaternion(qw,qx,qy,qz), gtsam.Point3(x,y,z))
                vals.insert(k, pose)
            elif tag == "VEC3":
                vx,vy,vz = map(float, parts[2:5])
                vals.insert(k, np.array([vx,vy,vz], dtype=float))
            elif tag == "POINT3":
                ax,ay,az = map(float, parts[2:5])
                vals.insert(k, gtsam.Point3(ax,ay,az))
            elif tag == "BIAS":
                ax,ay,az,gx,gy,gz = map(float, parts[2:8])
                vals.insert(k, gtsam.imuBias.ConstantBias(gtsam.Point3(ax,ay,az),
                                                          gtsam.Point3(gx,gy,gz)))
    return vals

# --------- save graph + values ---------------------------------------------
def save_graph_values(graph: gtsam.NonlinearFactorGraph,
                      values: gtsam.Values,
                      graph_path: str | os.PathLike,
                      values_path: str | os.PathLike) -> None:
    graph_path  = pathlib.Path(graph_path)
    values_path = pathlib.Path(values_path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    values_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Graphviz .graph with real keys and factor types preserved
    try:
        graph.saveGraph(str(graph_path), values, gtsam.DefaultKeyFormatter())
    except TypeError:
        # Older signature: (path, values) is enough
        graph.saveGraph(str(graph_path), values)

    # 2) Values
    if hasattr(values, "save"):
        values.save(str(values_path))
    else:
        _values_save_txt(values, values_path.with_suffix(".txt"))

# --------- export final ISAM2 estimates to CSV ------------------------------
def save_estimates(isam: gtsam.ISAM2,
                   csv_path: str | os.PathLike) -> None:
    """
    Write every Pose3 in ISAM2 to <csv_path>.

    Columns: key,sym,idx,x,y,z,qx,qy,qz,qw
    """
    csv_path = pathlib.Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    values = isam.calculateEstimate()
    if values.size() == 0:
        print("[save_estimates] WARNING – ISAM2 has no variables")
        return

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["key","sym","idx","x","y","z","qx","qy","qz","qw"])

        for k in values.keys():
            # Skip non‑pose variables
            try:
                pose: gtsam.Pose3 = values.atPose3(k)
            except RuntimeError:
                continue

            # --- extract symbol ------------------------------------------
            s = gtsam.Symbol(k)        # works irrespective of build
            try:                       # modern API
                _c = s.chr()           # may be int or str
                sym_char = chr(_c) if isinstance(_c, int) else _c
                sym_idx  = s.index()
            except AttributeError:     # very old API – fallback parse
                txt = str(s)           # 'x1' or 'Symbol(x,1)' …
                match = re.search(r"([A-Za-z])\s*,?\s*(\d+)", txt)
                if match is None:
                    sym_char, sym_idx = "?", 0
                else:
                    sym_char, sym_idx = match.group(1), int(match.group(2))
            # --------------------------------------------------------------

            # Handle gtsam.Pose3 objects with proper translation/rotation extraction
            try:
                # Get translation and rotation from the pose
                translation = pose.translation()
                rotation = pose.rotation()
                
                # Handle translation - could be Point3 or numpy array
                if hasattr(translation, 'x'):
                    # It's a Point3 with x(), y(), z() methods
                    t_x, t_y, t_z = translation.x(), translation.y(), translation.z()
                else:
                    # It's a numpy array or similar
                    t_x, t_y, t_z = float(translation[0]), float(translation[1]), float(translation[2])
                
                # Handle rotation - convert to quaternion
                quaternion = rotation.toQuaternion()
                if hasattr(quaternion, 'x'):
                    # It's a Quaternion with x(), y(), z(), w() methods
                    q_x, q_y, q_z, q_w = quaternion.x(), quaternion.y(), quaternion.z(), quaternion.w()
                else:
                    # It's a numpy array or similar
                    q_x, q_y, q_z, q_w = float(quaternion[0]), float(quaternion[1]), float(quaternion[2]), float(quaternion[3])
                
                w.writerow([k, sym_char, sym_idx, t_x, t_y, t_z, q_x, q_y, q_z, q_w])
                
            except Exception as e:
                print(f"[save_estimates] Error processing pose {k}: {e}")
                continue

    print(f"[save_estimates] wrote {csv_path.resolve()}")