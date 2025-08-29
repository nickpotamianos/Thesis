# swarm_ml/measure_adapter.py
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
import numpy as np

try:
    from swarm_ml import los_classification as losmod  # optional
except Exception:
    try:
        import los_classification as losmod
    except Exception:
        losmod = None

@dataclass
class AdapterConfig:
    base_range_var: float = 0.35**2  # conservative default (m^2)
    min_reliability: float = 1e-3
    max_reliability: float = 1.0
    bias_clip: float = 1.0
    alpha: float = 3.0
    beta: float = -1.0
    # Innovation adaptation
    ema_alpha: float = 0.05        # EMA for whiteness
    min_scale: float = 0.25
    max_scale: float = 8.0

class MeasureAdapter:
    """
    Bias-correct + reliability-shape + (optional) innovation-driven R scaling.
    """
    def __init__(self, cfg: AdapterConfig = AdapterConfig(), bias_model=None):
        self.cfg = cfg
        self.bias_model = bias_model
        self._residual_memory: Dict[Tuple[str, str], float] = {}
        self._whiten_ema: Dict[Tuple[str, str], float] = {}
        self._rscale: Dict[Tuple[str, str], float] = {}

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
        key = (tracker_id, target_id)

        # Bias
        if self.bias_model is not None and features is not None:
            pred = self.bias_model.predict(features)
            bias = float(np.clip(pred, -self.cfg.bias_clip, self.cfg.bias_clip))
        else:
            bias = 0.0
        z_corr = float(z - bias)

        # Reliability
        if los_score is not None:
            rrel = self._sigmoid(self.cfg.alpha * float(los_score) + self.cfg.beta)
        else:
            rrel = 0.5
        rrel = float(np.clip(rrel, self.cfg.min_reliability, self.cfg.max_reliability))

        # Base variance + reliability shaping
        R_eff = self.cfg.base_range_var / max(rrel, 1e-6)

        # Innovation-driven scaling (applied from previous steps)
        scale = self._rscale.get(key, 1.0)
        R_eff *= float(np.clip(scale, self.cfg.min_scale, self.cfg.max_scale))

        meta = {"bias": bias, "reliability": rrel, "R_eff": R_eff, "z_in": z, "z_corr": z_corr, "scale": scale}
        return z_corr, R_eff, meta

    def update_from_innov(self, tracker_id: str, target_id: str, innov: Optional[float], S: Optional[float]):
        """
        After a filter update, call this with the scalar innovation and S.
        Adjust a per-link variance scaling to drive E[(nu^2)/S] -> 1.
        """
        if innov is None or S is None or S <= 0:
            return
        key = (tracker_id, target_id)
        whiten = float((innov * innov) / S)
        ema = self._whiten_ema.get(key, 1.0)
        ema = (1.0 - self.cfg.ema_alpha) * ema + self.cfg.ema_alpha * whiten
        self._whiten_ema[key] = ema
        # Scale future R by current EMA
        self._rscale[key] = float(np.clip(ema, self.cfg.min_scale, self.cfg.max_scale))
