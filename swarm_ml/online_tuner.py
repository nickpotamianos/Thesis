#/swarm_ml/online_tuner.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import math
import numpy as np

Link = Tuple[str, str]  # (tracker_id, target_id)


@dataclass
class OnlineAdaptConfig:
    # NIS EMA for measurement calibration
    ema_alpha: float = 0.05
    r_min_scale: float = 0.5
    r_max_scale: float = 10.0

    # Q (process) inflation control (optional)
    q_adapt: bool = True
    q_alpha: float = 0.02
    q_min_scale_xy: float = 0.6
    q_max_scale_xy: float = 3.0
    q_min_scale_z: float = 0.6
    q_max_scale_z: float = 3.0
    q_gain: float = 0.15      # how strongly NIS>1 pushes Q up when R already high
    nis_tol: float = 0.10     # deadband around 1.0

    # Gating acceptance control
    gate_adapt: bool = True
    gate_target_accept: float = 0.97
    gate_sigma_init: float = 3.0
    gate_sigma_min: float = 2.0
    gate_sigma_max: float = 4.0
    gate_gain: float = 0.05   # 0.02..0.10 is safe

    # CI weights (if FusionNet not used)
    ci_lambda: float = 0.7    # NIS weight slope
    ci_use_los: bool = True
    ci_los_alpha: float = 0.4 # exponent on LOS score


class OnlineTuner:
    """
    Causal, per-link self-calibration using innovation statistics (no GT).
    Maintains:
      - EMA of NIS per link -> R scaling
      - gate_sigma adaptation to maintain target acceptance
      - optional global Q scaling (xy, z) based on persistent miscalibration
    """

    def __init__(self, cfg: OnlineAdaptConfig = OnlineAdaptConfig()):
        self.cfg = cfg
        # Per-link stats
        self._ema_nis: Dict[Link, float] = {}
        self._r_scale: Dict[Link, float] = {}
        self._gate_sigma: Dict[Link, float] = {}
        self._acc_pass: Dict[Link, int] = {}
        self._acc_tot: Dict[Link, int] = {}

        # Global process noise scales (apply to all trackers/target model)
        self._q_scale_xy: float = 1.0
        self._q_scale_z: float = 1.0

    # ---------- public API ----------
    def get_r_scale(self, link: Link) -> float:
        return float(self._r_scale.get(link, 1.0))

    def get_gate_sigma(self, link: Link) -> float:
        return float(self._gate_sigma.get(link, self.cfg.gate_sigma_init))

    def get_q_scales(self) -> Tuple[float, float]:
        return float(self._q_scale_xy), float(self._q_scale_z)

    def suggest_ci_weights(self,
                           keys: list[str],
                           nis_recent: Dict[str, float],
                           los_recent: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Optional: CI weights from recent per-tracker NIS and LOS (causal).
        w_i ∝ exp(-λ NIS_i) * LOS_i^α
        """
        lam = self.cfg.ci_lambda
        alpha = self.cfg.ci_los_alpha
        s = []
        for k in keys:
            nis_i = max(1e-6, float(nis_recent.get(k, 1.0)))
            w = math.exp(-lam * nis_i)
            if self.cfg.ci_use_los and los_recent is not None:
                los = float(np.clip(los_recent.get(k, 0.5), 0.05, 0.95))
                w *= (los ** alpha)
            s.append(w)
        s = np.asarray(s, dtype=float)
        if s.sum() <= 0:
            return {k: 1.0/len(keys) for k in keys}
        s /= s.sum()
        return {k: float(s[i]) for i, k in enumerate(keys)}

    # ---------- online updates ----------
    def after_gating(self, link: Link, accepted: bool):
        """Call once per attempted measurement, before EKF update."""
        self._acc_tot[link] = self._acc_tot.get(link, 0) + 1
        if accepted:
            self._acc_pass[link] = self._acc_pass.get(link, 0) + 1
        # adapt gate sigma occasionally
        if self.cfg.gate_adapt and self._acc_tot[link] % 100 == 0:
            self._adapt_gate_sigma(link)

    def after_update(self,
                     link: Link,
                     innovation: float,
                     S_scalar: float,
                     r_is_maxed: bool,
                     geom_ez: Optional[float] = None):
        """
        Call once for each USED measurement (i.e., after gating and EKF update).
        Updates EMA NIS -> R scaling. Optionally nudges Q if R is saturated.
        """
        if not np.isfinite(S_scalar) or S_scalar <= 0:
            return
        nis = float((innovation * innovation) / S_scalar)
        ema = self._ema_nis.get(link, 1.0)
        alpha = self.cfg.ema_alpha
        ema = (1.0 - alpha) * ema + alpha * nis
        self._ema_nis[link] = ema

        # R scaling tracks EMA NIS directly (bounded)
        r_scale = self._r_scale.get(link, 1.0)
        # smooth multiplicative update
        r_scale = r_scale ** (1.0 - alpha) * (ema ** alpha)
        r_scale = float(np.clip(r_scale, self.cfg.r_min_scale, self.cfg.r_max_scale))
        self._r_scale[link] = r_scale

        # Optional: Q inflation when NIS remains high AND R is already large
        if self.cfg.q_adapt:
            self._maybe_adapt_Q(ema_nis=ema, r_is_maxed=r_is_maxed, geom_ez=geom_ez)

    # ---------- internals ----------
    def _adapt_gate_sigma(self, link: Link):
        tgt = self.cfg.gate_target_accept
        acc = self._acc_pass.get(link, 0)
        tot = self._acc_tot.get(link, 1)
        rate = acc / max(1, tot)  # acceptance rate
        sigma = self._gate_sigma.get(link, self.cfg.gate_sigma_init)
        # multiplicative update toward target acceptance
        # if rate < tgt → increase sigma; else decrease
        delta = self.cfg.gate_gain * (tgt - rate)
        sigma = sigma * math.exp(delta)
        sigma = float(np.clip(sigma, self.cfg.gate_sigma_min, self.cfg.gate_sigma_max))
        self._gate_sigma[link] = sigma
        # reset counters to avoid drift
        self._acc_pass[link] = 0
        self._acc_tot[link] = 0

    def _maybe_adapt_Q(self, ema_nis: float, r_is_maxed: bool, geom_ez: Optional[float]):
        """
        Conservative covariance-matching:
        - If EMA NIS >> 1 and R already near max, increase Q a bit.
        - If EMA NIS << 1 and R not deflated, decrease Q a bit.
        ez (|bearing_z|) can bias z vs xy scaling.
        """
        tol = self.cfg.nis_tol
        # decide which axis to emphasize (geometry helps separate z from xy)
        ez = float(np.clip(geom_ez if geom_ez is not None else 0.5, 0.0, 1.0))
        bias_z = (ez > 0.7)  # high vertical observability → adjust z more

        if ema_nis > (1.0 + tol) and r_is_maxed:
            k = self.cfg.q_gain * (ema_nis - 1.0)
            if bias_z:
                self._q_scale_z = float(np.clip(self._q_scale_z * (1.0 + self.cfg.q_alpha * k),
                                                self.cfg.q_min_scale_z, self.cfg.q_max_scale_z))
            else:
                self._q_scale_xy = float(np.clip(self._q_scale_xy * (1.0 + self.cfg.q_alpha * k),
                                                 self.cfg.q_min_scale_xy, self.cfg.q_max_scale_xy))

        elif ema_nis < (1.0 - tol) and not r_is_maxed:
            k = self.cfg.q_gain * (1.0 - ema_nis)
            if bias_z:
                self._q_scale_z = float(np.clip(self._q_scale_z * (1.0 - self.cfg.q_alpha * k),
                                                self.cfg.q_min_scale_z, self.cfg.q_max_scale_z))
            else:
                self._q_scale_xy = float(np.clip(self._q_scale_xy * (1.0 - self.cfg.q_alpha * k),
                                                 self.cfg.q_min_scale_xy, self.cfg.q_max_scale_xy))

