import numpy as np
from typing import Dict

def project_to_safe(p_self: np.ndarray, move: np.ndarray, neighbors: Dict[str, np.ndarray], d_min: float = 1.5) -> np.ndarray:
    """
    Project a proposed move to maintain minimum separation from neighbors.
    If any neighbor would be closer than d_min after move, push away along the
    gradient direction. Simple, fast, and conservative.
    """
    if not neighbors:
        return move
    p_self = np.asarray(p_self, float).reshape(3)
    mv = np.asarray(move, float).reshape(3)
    grad = np.zeros(3, dtype=float)
    for _, pn in neighbors.items():
        pn = np.asarray(pn, float).reshape(3)
        rel = (p_self + mv) - pn
        d = float(np.linalg.norm(rel) + 1e-9)
        if d < d_min:
            grad += (rel / d) * (d_min - d)
    if np.linalg.norm(grad) > 0:
        mv = mv + 0.5 * grad
    return mv

