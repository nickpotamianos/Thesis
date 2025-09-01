# swarm_ml/distrib_ci.py
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np

@dataclass
class CommsConfig:
    rounds: int = 1        # consensus rounds per timestep
    p_link: float = 1.0    # probability an undirected link exists
    p_drop: float = 0.0    # independent packet drop per edge per round
    seed: int = 0

def _metropolis_weights(A: np.ndarray) -> np.ndarray:
    """Build row-stochastic Metropolis-Hastings weights for undirected graph A."""
    N = A.shape[0]
    deg = A.sum(axis=1)
    W = np.zeros_like(A, dtype=float)
    for i in range(N):
        for j in range(N):
            if i != j and A[i, j] > 0:
                W[i, j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        W[i, i] = 1.0 - W[i].sum()
    return W

class GossipFuser:
    """
    Decentralized CI via gossip on information form.
    Each node holds J_i, h_i; at each round it averages with neighbors using W (row-stochastic).
    Returns the network-averaged fused estimate (what a central logger would see).
    """
    def __init__(self, cfg: CommsConfig = CommsConfig()):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    def _rand_graph(self, N: int) -> np.ndarray:
        if self.cfg.p_link >= 1.0:
            A = np.ones((N, N)) - np.eye(N)
        else:
            U = self.rng.random((N, N))
            A = ((U < self.cfg.p_link).astype(float) * (1.0 - np.eye(N)))
            A = np.maximum(A, A.T)  # undirected
        return A

    def fuse(self, parts: Dict[str, Tuple[np.ndarray, np.ndarray]], weights: Optional[Dict[str, float]] = None):
        keys: List[str] = list(parts.keys())
        N = len(keys)
        if N == 1:
            mu, P = parts[keys[0]]
            return mu, P, {keys[0]: 1.0}

        # Build per-node information (CI-friendly jitter)
        J = []
        h = []
        for rid in keys:
            mu, P = parts[rid]
            Ji = np.linalg.inv(P + 1e-9 * np.eye(P.shape[0]))
            hi = Ji @ mu
            wi = 1.0 if (weights is None) else float(weights.get(rid, 1.0))
            J.append(wi * Ji); h.append(wi * hi)
        J = np.stack(J, axis=0)  # (N, d, d)
        h = np.stack(h, axis=0)  # (N, d)

        # Random communication graph and weights
        A = self._rand_graph(N)
        W = _metropolis_weights(A)

        # Perform consensus rounds with independent packet drops
        Jv = J.copy()
        hv = h.copy()
        for _ in range(self.cfg.rounds):
            # Drop edges independently (row-wise renormalization keeps convexity)
            if self.cfg.p_drop > 0.0:
                mask = (self.rng.random(W.shape) >= self.cfg.p_drop).astype(float)
                W_eff = W * mask
                row_sums = W_eff.sum(axis=1, keepdims=True)
                row_sums[row_sums == 0.0] = 1.0
                W_eff = W_eff / row_sums
            else:
                W_eff = W
            # Apply averaging: (N,N) @ (N,d,d) -> (N,d,d) via einsum; same for h
            Jv = np.einsum("ij,jkl->ikl", W_eff, Jv)
            hv = np.einsum("ij,jk->ik",   W_eff, hv)

        # What a central observer would see (nodes have converged if rounds are enough)
        J_bar = Jv.mean(axis=0)
        h_bar = hv.mean(axis=0)
        # Add regularization for numerical stability
        J_bar_reg = J_bar + 1e-6 * np.eye(J_bar.shape[0])
        P = np.linalg.inv(J_bar_reg)
        mu = P @ h_bar

        # Provide implicit weights (uniform if none were provided)
        if weights is None:
            w = {keys[i]: 1.0 / N for i in range(N)}
        else:
            # Normalize provided weights to sum to 1 for logging
            s = sum(max(0.0, float(weights.get(k, 0.0))) for k in keys)
            if s <= 0:
                w = {k: 1.0 / N for k in keys}
            else:
                w = {k: float(weights.get(k, 0.0)) / s for k in keys}
        return mu, P, w
