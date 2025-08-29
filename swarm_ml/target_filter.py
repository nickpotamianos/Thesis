# swarm_ml/target_filter.py
from dataclasses import dataclass
from typing import Tuple, Optional, Dict, Any
import numpy as np
from numpy.linalg import inv, slogdet

@dataclass
class IFConfig:
    sigma_a_xy: float = 1.0     # horiz accel noise std (m/s^2)
    sigma_a_z: float  = 0.5     # vertical accel noise std (m/s^2) -> tighter by default
    p0_xy: float = 2.0          # initial pos std (x,y)
    p0_z:  float = 2.0          # initial pos std (z)
    v0_xy: float = 1.0          # initial vel std (x,y)
    v0_z:  float = 1.0          # initial vel std (z)
    gate_N_sigma: float = 3.0

class TargetIF:
    """
    3D constant-velocity target filter in information form:
      x = [p(3), v(3)], F = [[I, dt*I],[0,I]]
      Q(dt) uses per-axis acceleration noise (sigma_a_xy, sigma_a_z).
    Measurement: z = ||p - p_tracker|| + noise.
    """
    def __init__(self, x0: Optional[np.ndarray] = None, cfg: IFConfig = IFConfig()):
        self.cfg = cfg
        if x0 is None:
            self.mu = np.zeros(6)
        else:
            self.mu = x0.reshape(6)

        P0 = np.diag([
            cfg.p0_xy**2, cfg.p0_xy**2, cfg.p0_z**2,
            cfg.v0_xy**2, cfg.v0_xy**2, cfg.v0_z**2
        ])
        self.J = inv(P0)
        self.h = self.J @ self.mu

    @staticmethod
    def _F(dt: float) -> np.ndarray:
        I = np.eye(3)
        return np.block([[I, dt*I],
                         [np.zeros((3,3)), I]])

    def _Q(self, dt: float) -> np.ndarray:
        # axis-wise CV Q
        qx = self.cfg.sigma_a_xy**2
        qy = self.cfg.sigma_a_xy**2
        qz = self.cfg.sigma_a_z**2
        Q_axis = lambda q: np.block([
            [ (dt**3)/3 * q, (dt**2)/2 * q ],
            [ (dt**2)/2 * q,    dt * q     ]
        ])
        # build per-axis then assemble
        Qx = Q_axis(qx); Qy = Q_axis(qy); Qz = Q_axis(qz)
        Q = np.zeros((6,6))
        Q[np.ix_([0,3],[0,3])] = Qx
        Q[np.ix_([1,4],[1,4])] = Qy
        Q[np.ix_([2,5],[2,5])] = Qz
        return Q

    def predict(self, dt: float):
        F = self._F(dt)
        Q = self._Q(dt)
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
        h0, H = self._range_linearize(self.mu, tracker_pos)
        S = H @ inv(self.J) @ H.T + R
        innov = z - h0
        if float(innov**2 / S) > self.cfg.gate_N_sigma**2:
            return {"used": False, "innov": float(innov), "S": float(S)}

        J_meas = (1.0/R) * (H.T @ H)
        h_meas = (1.0/R) * (H.T @ (z - h0 + H @ self.mu))

        self.J = self.J + J_meas
        self.h = self.h + h_meas
        self.mu = inv(self.J) @ self.h
        return {"used": True, "innov": float(innov), "S": float(S), "h0": float(h0)}

    def posterior(self) -> Tuple[np.ndarray, np.ndarray]:
        P = inv(self.J)
        return self.mu.copy(), P

    @staticmethod
    def cov_metrics(P: np.ndarray) -> Dict[str, float]:
        sign, logdet = slogdet(P)
        return {"tr": float(np.trace(P)), "logdet": float(logdet)}
