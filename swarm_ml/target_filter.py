# swarm_ml/target_filter.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from numpy.linalg import inv, slogdet

@dataclass
class IFConfig:
    sigma_a: float = 1.0    # process accel noise std (m/s^2)
    p0: float = 2.0         # initial position std (m)
    v0: float = 1.0         # initial velocity std (m/s)
    gate_N_sigma: float = 4.0

class TargetIF:
    """
    3D constant-velocity target filter in information form:
      x = [p(3), v(3)], F = [[I, dt*I],[0,I]], Q(dt) standard CV.
    Measurement: z = ||p - p_tracker|| + noise.
    """
    def __init__(self, x0: Optional[np.ndarray] = None, cfg: IFConfig = IFConfig()):
        self.cfg = cfg
        if x0 is None:
            self.mu = np.zeros(6)
        else:
            self.mu = x0.reshape(6)
        P0 = np.diag([cfg.p0**2]*3 + [cfg.v0**2]*3)
        self.J = inv(P0)
        self.h = self.J @ self.mu

        self._last_lin_point: Optional[np.ndarray] = None  # for diagnostics

    @staticmethod
    def _F_Q(dt: float, sigma_a: float) -> Tuple[np.ndarray, np.ndarray]:
        I = np.eye(3)
        F = np.block([[I, dt*I],
                      [np.zeros((3,3)), I]])
        q = sigma_a**2
        Q = np.block([[ (dt**3)/3 * q * I, (dt**2)/2 * q * I],
                      [ (dt**2)/2 * q * I,    dt * q * I    ]])
        return F, Q

    def predict(self, dt: float):
        F, Q = self._F_Q(dt, self.cfg.sigma_a)
        P = inv(self.J)
        mu = self.mu

        mu_pred = F @ mu
        P_pred = F @ P @ F.T + Q

        self.J = inv(P_pred)
        self.h = self.J @ mu_pred
        self.mu = mu_pred

    @staticmethod
    def _range_linearize(mu: np.ndarray, tracker_pos: np.ndarray) -> Tuple[float, np.ndarray]:
        p = mu[:3].reshape(3)
        diff = p - tracker_pos.reshape(3)
        d = float(np.linalg.norm(diff) + 1e-12)
        H = np.zeros((1, 6))
        H[0, :3] = (diff / d)
        h0 = d
        return h0, H

    def correct(self, z: float, R: float, tracker_pos: np.ndarray) -> Dict[str, Any]:
        # Gate
        h0, H = self._range_linearize(self.mu, tracker_pos)
        S = H @ inv(self.J) @ H.T + R
        innov = z - h0
        if float(innov**2 / S) > self.cfg.gate_N_sigma**2:
            return {"used": False, "innov": innov, "S": float(S)}

        # Info update (linearized)
        J_meas = H.T @ (1.0/R) @ H
        h_meas = H.T @ (1.0/R) * (z - h0 + H @ self.mu)  # standard IF linearized form

        self.J = self.J + J_meas
        self.h = self.h + h_meas
        self.mu = inv(self.J) @ self.h

        return {"used": True, "innov": float(innov), "S": float(S), "h0": float(h0)}

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        P = inv(self.J)
        return self.mu.copy(), P

    @staticmethod
    def info_from_mean_cov(mu: np.ndarray, P: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        J = inv(P)
        h = J @ mu
        return J, h

    @staticmethod
    def cov_metrics(P: np.ndarray) -> Dict[str, float]:
        sign, logdet = slogdet(P)
        return {"tr": float(np.trace(P)), "logdet": float(logdet)}
