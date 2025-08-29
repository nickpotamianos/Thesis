# swarm_ml/features.py
from typing import Dict, Any, Optional
import numpy as np

def se_translation_from_matrix(T: np.ndarray) -> np.ndarray:
    """
    Extract 3D translation from a 4x4 SE(3) or 5x5 SE_2(3) matrix.
    Falls back to last column's first 3 entries.
    """
    T = np.asarray(T)
    if T.shape == (4, 4) or T.shape == (5, 5):
        return T[:3, -1].reshape(3)
    if T.size >= 3:
        return np.array(T).reshape(-1)[:3]
    raise ValueError("Unknown pose format for extracting translation.")

def build_measurement_features(tracker_pos: np.ndarray,
                               target_pred_pos: Optional[np.ndarray],
                               uwb_range: float,
                               los_score: Optional[float],
                               residual_hist: Optional[np.ndarray] = None,
                               height_tracker: Optional[float] = None,
                               height_target: Optional[float] = None) -> np.ndarray:
    """
    Create a feature vector for ML-based measurement quality/bias modeling.
    """
    if target_pred_pos is None:
        geom = np.zeros(4)
    else:
        diff = target_pred_pos - tracker_pos
        dist = np.linalg.norm(diff) + 1e-9
        bearing = diff / dist
        geom = np.hstack([bearing, dist])

    f = [uwb_range]
    f.extend(geom.tolist())
    f.append(los_score if los_score is not None else 0.5)

    if height_tracker is not None and height_target is not None:
        f.append(height_target - height_tracker)
    else:
        f.append(0.0)

    if residual_hist is not None and residual_hist.size > 0:
        f.extend([np.mean(residual_hist), np.var(residual_hist)])
    else:
        f.extend([0.0, 0.0])
    return np.asarray(f, dtype=float)
