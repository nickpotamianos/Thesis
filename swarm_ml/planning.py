# swarm_ml/planning.py
import numpy as np
from typing import Dict, Tuple, List

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
        vz = np.array([0.0, 0.0, np.sign(b[2]) or 1.0])
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
        moves[k] = move
    return moves