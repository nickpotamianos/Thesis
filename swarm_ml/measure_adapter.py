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
    # New: LOS/geometry influences
    los_influence: float = 0.5     # controls how strongly LOS adjusts reliability
    geom_influence: float = 0.5    # controls how strongly |e_z| adjusts reliability

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
        # Online bias estimator per (tracker,target) link
        self._bias_ema: Dict[Tuple[str, str], float] = {}
        self._bias_beta: float = 0.01  # slow learn-rate for constant bias
        self._bias_clip: float = float(self.cfg.bias_clip)

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

        # Bias: model-predicted + online EMA estimate (both gently clipped)
        bias_est = float(self._bias_ema.get(key, 0.0))
        bias_model = 0.0
        if self.bias_model is not None and features is not None:
            pred = self.bias_model.predict(features)
            bias_model = float(np.clip(pred, -self.cfg.bias_clip, self.cfg.bias_clip))
        bias = float(np.clip(bias_model + bias_est, -self._bias_clip, self._bias_clip))
        z_corr = float(z - bias)

        # Reliability from LOS score (prob in [0,1])
        if los_score is not None:
            ls = float(np.clip(float(los_score), 0.0, 1.0))
            # Map to a mild multiplier around 1.0
            los_mult = 1.0 + self.cfg.los_influence * (ls - 0.5)  # 0.75..1.25 if influence=0.5
        else:
            los_mult = 1.0

        # Reliability from geometry (|bearing_z|); small -> downweight, large -> upweight
        if target_pred_pos is not None and tracker_pos is not None:
            a = np.asarray(target_pred_pos, float) - np.asarray(tracker_pos, float)
            nrm = np.linalg.norm(a) + 1e-9
            ez = abs(a[2]) / nrm
            geom_mult = 1.0 + self.cfg.geom_influence * (ez - 0.5)  # ~0.75..1.25 typically
        else:
            geom_mult = 1.0

        # Compose reliability and clip
        rrel = float(np.clip(los_mult * geom_mult, self.cfg.min_reliability, self.cfg.max_reliability))

        # Base variance + reliability shaping
        R_eff = self.cfg.base_range_var / max(rrel, 1e-6)

        # Innovation-driven scaling (applied from previous steps)
        scale = self._rscale.get(key, 1.0)
        R_eff *= float(np.clip(scale, self.cfg.min_scale, self.cfg.max_scale))

        meta = {
            "bias": bias,
            "bias_online": bias_est,
            "bias_model": bias_model,
            "reliability": rrel,
            "R_eff": R_eff,
            "z_in": z,
            "z_corr": z_corr,
            "scale": scale,
        }
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
        # Slowly adapt online bias toward mean innovation
        b = float(self._bias_ema.get(key, 0.0))
        b = (1.0 - self._bias_beta) * b + self._bias_beta * float(innov)
        self._bias_ema[key] = float(np.clip(b, -self._bias_clip, self._bias_clip))
