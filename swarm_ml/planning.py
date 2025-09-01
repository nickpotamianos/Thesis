# swarm_ml/planning.py
import numpy as np
from typing import Dict, Tuple, List
from .safety import project_to_safe
from .active_sensing import expected_trace_reduction

def suggest_vantage_moves(target_pos: np.ndarray,
                          tracker_pos: Dict[str, np.ndarray],
                          desired_range: Tuple[float, float] = (6.0, 12.0),
                          up_bias: float = 0.3) -> Dict[str, np.ndarray]:
    """
    Suggest a small move vector per tracker to increase information about z and maintain spacing.
    up_bias in [0,1]: how much to prefer vertical separation |b_z|.
    Returns: {tracker -> delta_xyz} (not scaled to dynamics; just a hint).
    """
    p_t = target_pos[:3].reshape(3)
    moves = {}
    # crude angular diversity: spread by azimuth
    ids = list(tracker_pos.keys())
    az = []
    for k in ids:
        v = p_t - tracker_pos[k].reshape(3)
        az.append(np.arctan2(v[1], v[0]))
    order = np.argsort(az)
    # neighbors in azimuth should separate
    for rank, idx in enumerate(order):
        k = ids[idx]
        p = tracker_pos[k].reshape(3)
        v = p_t - p
        d = np.linalg.norm(v) + 1e-9
        b = v / d
        # range term
        dmin, dmax = desired_range
        r_push = 0.0
        if d < dmin: r_push = -(dmin - d) / dmin
        elif d > dmax: r_push = (d - dmax) / dmax
        # vertical observability push: move to increase |b_z|
        # move opposite to target along z to increase |b_z|
        vz = np.array([0.0, 0.0, -np.sign(b[2]) if b[2] != 0 else -1.0])
        # azimuthal separation (repel from neighbors)
        left = ids[order[(rank - 1) % len(ids)]]
        right = ids[order[(rank + 1) % len(ids)]]
        repel = 0.0
        for nbr in [left, right]:
            vn = p_t - tracker_pos[nbr].reshape(3)
            repel += np.cross(vn, np.array([0,0,1.0]))[:2] @ np.cross(v, np.array([0,0,1.0]))[:2]
        repel_dir = np.array([-b[1], b[0], 0.0]) * np.sign(repel)
        # combine
        move = (1.0 - up_bias) * (r_push * b + 0.1 * repel_dir) + up_bias * vz * 0.5
        # project to maintain separation (simple barrier)
        nbrs = {n: tracker_pos[n] for n in ids if n != k}
        move = project_to_safe(p, move, nbrs, d_min=1.5)
        moves[k] = move
    return moves


def suggest_vantage_moves_eig(mu_star: np.ndarray,
                              P_star: np.ndarray,
                              tracker_pos: Dict[str, np.ndarray],
                              r_eff_map: Dict[str, float],
                              desired_range: Tuple[float, float] = (6.0, 12.0),
                              step: float = 0.5) -> Dict[str, np.ndarray]:
    """
    EIG-based myopic planner: pick among candidate directions the one maximizing
    expected trace reduction, with simple range constraint shaping and separation.
    """
    moves: Dict[str, np.ndarray] = {}
    p_t = mu_star[:3].reshape(3)
    # candidate unit directions
    dirs = [
        np.array([1,0,0], float), np.array([-1,0,0], float),
        np.array([0,1,0], float), np.array([0,-1,0], float),
        np.array([0,0,1], float), np.array([0,0,-1], float),
    ]
    for k, p in tracker_pos.items():
        p = p.reshape(3)
        b = (p_t - p); d = float(np.linalg.norm(b) + 1e-9)
        b = b / d
        dirs_ext = dirs + [b, -b]
        best_dir = np.zeros(3)
        best_score = -1e18
        R_eff = float(r_eff_map.get(k, 0.35**2))
        for dvec in dirs_ext:
            cand_p = p + step * dvec
            eig = expected_trace_reduction(mu_star, P_star, cand_p, R_eff)
            # range shaping
            dmin, dmax = desired_range
            dn = float(np.linalg.norm(p_t - cand_p))
            range_pen = 0.0
            if dn < dmin: range_pen = (dmin - dn)
            elif dn > dmax: range_pen = (dn - dmax)
            score = eig - 0.5 * range_pen
            if score > best_score:
                best_score = score
                best_dir = dvec
        move = best_dir * step
        # safety projection
        nbrs = {n: tracker_pos[n] for n in tracker_pos.keys() if n != k}
        move = project_to_safe(p, move, nbrs, d_min=1.5)
        moves[k] = move
    return moves
