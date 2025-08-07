# -*- coding: utf-8 -*-
"""
Light‑weight IO helpers for factor‑graph experiments.
"""

from __future__ import annotations
import os, csv, re, pathlib, gtsam

# --------------------------------------------------------------------------- #
def ensure_dir(path: str | os.PathLike) -> str:
    path = str(path)
    os.makedirs(path, exist_ok=True)
    return path

# --------------------------------------------------------------------------- #
def save_graph_values(graph: gtsam.NonlinearFactorGraph,
                     values: gtsam.Values,
                     graph_path: str | os.PathLike,
                     values_path: str | os.PathLike) -> None:
    """
    Save a NonlinearFactorGraph and Values to files that can be loaded later.
    """
    graph_path = pathlib.Path(graph_path)
    values_path = pathlib.Path(values_path)
    graph_path.parent.mkdir(parents=True, exist_ok=True)
    values_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1) factors *and* initial values in one .g2o file
    #    (writeG2o serialises both)
    gtsam.writeG2o(graph, values, str(graph_path))

    # 2) optionally store the Values as a plain text file –
    #    only works when Boost‑serialization wrappers are present.
    try:                                # modern wheels built WITH support
        values.save(str(values_path))
    except AttributeError:              # most wheels – feature not compiled
        # Either ignore, or fall back to numpy/pickle/etc.  For now we skip.
        pass

# --------------------------------------------------------------------------- #
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
