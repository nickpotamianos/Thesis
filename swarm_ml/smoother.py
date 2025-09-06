# swarm_ml/smoother.py
from dataclasses import dataclass
import numpy as np

@dataclass
class CVNoise:
    sigma_a_xy: float
    sigma_a_z: float

def F_mat(dt: float):
    """Constant velocity transition matrix"""
    I = np.eye(3)
    return np.block([[I, dt*I],
                     [np.zeros((3,3)), I]])

def Q_mat(dt: float, noise: CVNoise):
    """Constant velocity process noise matrix"""
    qx = noise.sigma_a_xy**2
    qy = noise.sigma_a_xy**2
    qz = noise.sigma_a_z**2

    def Q1(q):
        return np.array([[ (dt**3)/3*q, (dt**2)/2*q ],
                         [ (dt**2)/2*q,    dt*q    ]], dtype=float)

    Q = np.zeros((6,6))
    Q[np.ix_([0,3],[0,3])] = Q1(qx)
    Q[np.ix_([1,4],[1,4])] = Q1(qy)
    Q[np.ix_([2,5],[2,5])] = Q1(qz)
    return Q

def rts_smooth(mu_f, P_f, dts, noise: CVNoise):
    """
    Rauch-Tung-Striebel smoothing for constant velocity model.
    mu_f, P_f: lists/arrays of filtered means and covariances (after fusion)
    dts: list of time steps
    Returns: smoothed means and covariances
    """
    N = len(mu_f)
    mu_s = [m.copy() for m in mu_f]
    P_s  = [S.copy() for S in P_f]

    # Normalize dts: accept either length N-1 (preferred) or length N with dts[0]=0
    dts = np.asarray(dts).reshape(-1)
    if dts.size == N - 1:
        def _dt_at(k): return float(dts[k])
    elif dts.size == N:
        def _dt_at(k): return float(dts[k+1])
    else:
        raise ValueError(f"dts length must be N-1 or N; got {dts.size} for N={N}")

    # Forward pass: store predictions
    mu_p = [None]*N
    P_p  = [None]*N
    for k in range(N-1):
        dt = _dt_at(k)
        F = F_mat(dt)
        Q = Q_mat(dt, noise)
        mu_p[k+1] = F @ mu_f[k]
        P_p[k+1]  = F @ P_f[k] @ F.T + Q

    # Backward pass
    for k in range(N-2, -1, -1):
        dt = _dt_at(k)
        F  = F_mat(dt)
        # stable: solve instead of inverse
        # Ck = P_k F^T (P_k+1|k)^-1  ->  solve(P_p, (F @ P_f[k]).T).T
        FPk = F @ P_f[k]
        Ck  = np.linalg.solve(P_p[k+1].T, FPk.T).T
        mu_s[k] = mu_s[k] + Ck @ (mu_s[k+1] - mu_p[k+1])
        P_s[k]  = P_s[k] + Ck @ (P_s[k+1] - P_p[k+1]) @ Ck.T
        # light symmetrization for numerical hygiene
        P_s[k]  = 0.5 * (P_s[k] + P_s[k].T)

    return np.array(mu_s), np.array(P_s)