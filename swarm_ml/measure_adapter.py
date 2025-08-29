# swarm_ml/measure_adapter.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    # Prefer a local copy under swarm_ml if you made one
    from swarm_ml import los_classification as losmod  # type: ignore
except Exception:
    try:
        import los_classification as losmod  # repo root
    except Exception:
        losmod = None

@dataclass
class AdapterConfig:
    base_range_var: float = 0.10**2  # (m^2) nominal UWB variance
    min_reliability: float = 1e-3
    max_reliability: float = 1.0
    bias_clip: float = 1.0  # cap bias correction magnitude
    # Reliability shaping: r = sigmoid(alpha * s + beta)
    alpha: float = 3.0
    beta: float = -1.0

class MeasureAdapter:
    """
    Produces (bias-corrected) range and an effective measurement covariance using
    (optional) LOS/NLOS inference signal and residual heuristics. Plug-in point for BiasNet.
    """
    def __init__(self, cfg: AdapterConfig = AdapterConfig(), bias_model=None):
        self.cfg = cfg
        self.bias_model = bias_model
        self._residual_memory: Dict[Tuple[str, str], float] = {}

    @staticmethod
    def _sigmoid(x: float) -> float:
        return 1. / (1. + np.exp(-x))

    def correct(self,
                tracker_id: str,
                target_id: str,
                z: float,
                tracker_pos: np.ndarray,
                target_pred_pos: Optional[np.ndarray],
                los_score: Optional[float],
                features: Optional[np.ndarray] = None) -> Tuple[float, float, Dict[str, Any]]:
        """
        Returns:
          z_corr: corrected range
          R_eff:  effective variance
          meta:   dict with diagnostics
        """
        # Bias from learned model if provided
        if self.bias_model is not None and features is not None:
            pred = self.bias_model.predict(features)  # must return scalar bias
            bias = float(np.clip(pred, -self.cfg.bias_clip, self.cfg.bias_clip))
        else:
            bias = 0.0

        z_corr = float(z - bias)

        # Reliability from LOS classifier (if present) or from geometry
        if los_score is not None:
            r = self._sigmoid(self.cfg.alpha * float(los_score) + self.cfg.beta)
        else:
            # fallback: slightly conservative default
            r = 0.5

        r = float(np.clip(r, self.cfg.min_reliability, self.cfg.max_reliability))

        # Effective variance decreases with reliability
        R_eff = self.cfg.base_range_var / max(r, 1e-6)

        meta = {"bias": bias, "reliability": r, "R_eff": R_eff, "z_in": z, "z_corr": z_corr}
        return z_corr, R_eff, meta
