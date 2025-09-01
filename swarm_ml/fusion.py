# swarm_ml/fusion.py
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
import numpy as np
from numpy.linalg import inv

@dataclass
class CIFuserConfig:
    objective: str = "logdet"   # "logdet" or "trace"
    grid_step: float = 0.1       # weight grid for small N; safe & reproducible

class CIFuser:
    """
    CI fusion over N tracker posteriors (mu_i, P_i). Supports:
      - uniform weights
      - grid-search weights to minimize trace/logdet(P)
      - learned weights via a model that outputs positive weights -> softmax
    """
    def __init__(self, cfg: CIFuserConfig = CIFuserConfig(), weight_model=None):
        self.cfg = cfg
        self.weight_model = weight_model

    @staticmethod
    def _fuse_given_weights(parts: Dict[str, Tuple[np.ndarray, np.ndarray]],
                            weights: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray]:
        J_sum = None
        h_sum = None
        for rid, (mu, P) in parts.items():
            w = weights[rid]
            # numerical stability jitter
            J_i = inv(P + 1e-9 * np.eye(P.shape[0], dtype=P.dtype))
            h_i = J_i @ mu
            if J_sum is None:
                J_sum = w * J_i
                h_sum = w * h_i
            else:
                J_sum += w * J_i
                h_sum += w * h_i
        P = inv(J_sum)
        mu = P @ h_sum
        return mu, P

    @staticmethod
    def _uniform_weights(keys: List[str]) -> Dict[str, float]:
        w = 1.0 / max(1, len(keys))
        return {k: w for k in keys}

    @staticmethod
    def _simplex_grid(n: int, step: float) -> np.ndarray:
        """
        Generate simplex points of dimension n that sum to 1 with given step.
        """
        if n == 1:
            return np.array([[1.0]])
        # recursion by stars-and-bars enumeration
        levels = int(round(1.0 / step))
        grids = []
        def rec(prefix, remain, depth):
            if depth == n-1:
                grids.append(prefix + [remain])
                return
            for i in range(remain+1):
                rec(prefix + [i], remain - i, depth + 1)
        rec([], levels, 0)
        arr = np.array(grids, dtype=float) / levels
        return arr

    def _objective(self, P: np.ndarray) -> float:
        if self.cfg.objective == "trace":
            return float(np.trace(P))
        # default: logdet
        sign, logdet = np.linalg.slogdet(P)
        return float(logdet)

    def fuse(self,
             parts: Dict[str, Tuple[np.ndarray, np.ndarray]],
             method: str = "uniform",
             node_features: Optional[Dict[str, np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        keys = list(parts.keys())
        if len(keys) == 1:
            mu, P = parts[keys[0]]
            return mu, P, {keys[0]: 1.0}

        if method == "uniform":
            w = self._uniform_weights(keys)
            mu, P = self._fuse_given_weights(parts, w)
            return mu, P, w

        if method == "learned" and self.weight_model is not None and node_features is not None:
            # Stack node features in the same fusion order and get normalized weights
            X = np.vstack([node_features[k].reshape(1, -1) for k in keys])
            weights_vec = self.weight_model.predict_weights(X)  # (N,)
            # Guard rails: clip and renormalize
            weights_vec = np.maximum(weights_vec, 1e-9)
            weights_vec = weights_vec / np.sum(weights_vec)
            w = {k: float(weights_vec[i]) for i, k in enumerate(keys)}
            mu, P = self._fuse_given_weights(parts, w)
            return mu, P, w

        # Grid-search CI weights (robust and dependency-free)
        grid = self._simplex_grid(len(keys), self.cfg.grid_step)
        best_val = np.inf
        best_w = None
        best_mu, best_P = None, None
        for g in grid:
            w = {k: float(g[i]) for i, k in enumerate(keys)}
            if abs(sum(w.values()) - 1.0) > 1e-9:
                continue
            mu, P = self._fuse_given_weights(parts, w)
            val = self._objective(P)
            if val < best_val:
                best_val = val
                best_w = w
                best_mu, best_P = mu, P
        return best_mu, best_P, best_w
