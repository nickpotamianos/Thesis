import numpy as np
from .target_filter import TargetIF

def expected_trace_reduction(mu: np.ndarray, P: np.ndarray, tracker_p: np.ndarray, R: float) -> float:
    """
    approximate Δtr(P) from adding a single range at tracker_p
    """
    h0, H = TargetIF._range_linearize(mu, tracker_p)
    S = H @ P @ H.T + R
    J_add = (H.T @ H) / float(S)
    P_new = P - P @ J_add @ P
    return float(np.trace(P) - np.trace(P_new))

