#swarm_ml/active_sensing.py
import numpy as np
from .target_filter import TargetIF

def expected_trace_reduction(mu: np.ndarray, P: np.ndarray,
                             tracker_p: np.ndarray, R: float) -> float:
    """
    One-step A-optimal (trace) reduction for a scalar range measurement
    linearized at (mu, tracker_p).
    """
    _, H = TargetIF._range_linearize(mu, tracker_p)
    S = float(H @ P @ H.T + R)  # scalar
    KPH = (P @ H.T) * (1.0 / S)  # (6,1)
    P_new = P - KPH @ (H @ P)    # Joseph not needed for scalar, deterministic linearization
    return float(np.trace(P) - np.trace(P_new))

def batch_expected_trace_reduction(mu: np.ndarray, P: np.ndarray,
                                   trackers: np.ndarray, R: float) -> np.ndarray:
    return np.asarray([expected_trace_reduction(mu, P, p, R) for p in trackers], dtype=float)
