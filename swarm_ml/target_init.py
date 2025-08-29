# swarm_ml/target_init.py
from typing import Dict, Tuple, Optional
import numpy as np
import pandas as pd

def multilaterate_linear(p_list, z_list):
    # Linear LS via difference of sphere equations
    # choose first as reference
    p0 = p_list[0]; z0 = z_list[0]
    A=[]; b=[]
    for i in range(1, len(p_list)):
        pi = p_list[i]; zi = z_list[i]
        A.append(2*(pi - p0))
        b.append((np.dot(pi,pi) - np.dot(p0,p0)) - (zi*zi - z0*z0))
    A = np.vstack(A); b = np.array(b).reshape(-1,1)
    try:
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        return x.flatten()
    except Exception:
        return p0.copy()

def robust_multilateration(p_list, z_list, huber_delta=0.6, iters=15):
    # Initialize with linear solution, then Gauss-Newton with Huber
    p = multilaterate_linear(p_list, z_list)
    for _ in range(iters):
        r=[]; J=[]
        for pi, zi in zip(p_list, z_list):
            d = np.linalg.norm(p - pi) + 1e-12
            r.append(d - zi)
            J.append((p - pi)/d)
        r = np.array(r); J = np.vstack(J)
        w = np.ones_like(r)
        mask = np.abs(r) > huber_delta
        w[mask] = huber_delta/np.abs(r[mask])
        # weighted GN step
        W = np.diag(w)
        try:
            H = J.T @ W @ J
            g = J.T @ (W @ r)
            step = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            break
        p = p + step
        if np.linalg.norm(step) < 1e-4:
            break
    return p

def init_target_from_window(query_ts, uwb_range_df, tag_map, tracker_ids, target_id,
                            tracker_pose_fn, agg_fn, window_len=15):
    """
    Build a p0 from the earliest window where we have data.
    - tracker_pose_fn(trk, t) -> np.ndarray(3,) tracker sensor position in world
    - agg_fn(df_rows_for_trk_at_t) -> scalar range (e.g., robust_range_aggregate)
    Returns p0 or None if insufficient geometry.
    """
    # choose a start index that yields at least one measurement per tracker
    for start in range(0, min(window_len, len(query_ts))):
        end = min(start + window_len, len(query_ts))
        p_list=[]; z_list=[]
        for trk in tracker_ids:
            zs=[]
            ps=[]
            for t in query_ts[start:end]:
                df_t = uwb_range_df[uwb_range_df["timestamp"] == t]
                trk_tags = tag_map.get(trk, [])
                tgt_tags = tag_map.get(target_id, [])
                df_pair = df_t[
                    ((df_t["from_id"].isin(trk_tags)) & (df_t["to_id"].isin(tgt_tags))) |
                    ((df_t["from_id"].isin(tgt_tags)) & (df_t["to_id"].isin(trk_tags)))
                ]
                if df_pair.empty: continue
                z_t, _, _ = agg_fn(df_pair)
                ps.append(tracker_pose_fn(trk, t))
                zs.append(z_t)
            if zs:
                # use median across the window
                p_list.append(np.median(np.vstack(ps), axis=0))
                z_list.append(float(np.median(zs)))
        if len(p_list) >= 3:
            p0 = robust_multilateration(p_list, z_list)
            return p0
    return None
