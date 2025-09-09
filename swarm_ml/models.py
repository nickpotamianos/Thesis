# swarm_ml/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence

class BiasNet(nn.Module):
    """
    Small MLP that predicts scalar bias for a single UWB range measurement.
    Input features must match features.build_measurement_features(...).
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Optional feature normalization (mirrors FusionNet)
        self.register_buffer("x_mu",  torch.zeros(1, in_dim))
        self.register_buffer("x_std", torch.ones(1,  in_dim))
        self._use_norm = False

    def set_normalizer(self, mu, std) -> None:
        mu  = torch.as_tensor(mu,  dtype=torch.float32).reshape(1, -1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1)
        std = torch.clamp(std, min=1e-6)
        if mu.shape[1] != self.in_dim or std.shape[1] != self.in_dim:
            raise ValueError(f"BiasNet normalizer dim mismatch: expected {self.in_dim}, "
                             f"got mu={mu.shape[1]} std={std.shape[1]}")
        with torch.no_grad():
            self.x_mu.copy_(mu)
            self.x_std.copy_(std)
        self._use_norm = True

    def forward(self, x):
        if self._use_norm:
            x = (x - self.x_mu) / self.x_std
        return self.net(x).squeeze(-1)  # (B,)

    def predict(self, x_np):
        with torch.no_grad():
            x = torch.tensor(x_np, dtype=torch.float32).unsqueeze(0)
            y = self.forward(x).item()
        return y


class FusionNet(nn.Module):
    """
    Predicts per-tracker weights via attention-like scoring.
    Given per-node features (d-dimensional), produce normalized weights over nodes.
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.in_dim = int(in_dim)
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Dropout(p=0.10),
            nn.ReLU()
        )
        self.score = nn.Linear(hidden, 1)
        # Optional feature normalization (populated during training or by loader)
        self.register_buffer("x_mu",  torch.zeros(1, in_dim))
        self.register_buffer("x_std", torch.ones(1,  in_dim))
        self._use_norm = False

    def set_normalizer(self, mu: Sequence[float], std: Sequence[float]) -> None:
        """
        Install per-feature mean/std so training-time normalization is reproduced at inference.
        """
        mu = torch.as_tensor(mu, dtype=torch.float32).reshape(1, -1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, -1)
        std = torch.clamp(std, min=1e-6)
        if mu.shape[1] != self.in_dim or std.shape[1] != self.in_dim:
            raise ValueError(f"Normalizer dim mismatch: expected {self.in_dim}, got {mu.shape[1]}")
        with torch.no_grad():
            self.x_mu.copy_(mu)
            self.x_std.copy_(std)
        self._use_norm = True

    def forward(self, X):  # X: (N_nodes, d)
        if self._use_norm:
            X = (X - self.x_mu) / self.x_std
        H = self.enc(X)                   # (N,h)
        s = self.score(H).squeeze(-1)     # (N,)
        w = F.softmax(s, dim=0)           # (N,)
        return w

    def predict_weights(self, X_np):
        """
        X_np: array-like of shape (N_nodes, d)
        Returns: numpy array of shape (N_nodes,) that sums to 1.
        """
        self.eval()
        with torch.no_grad():
            X = torch.tensor(X_np, dtype=torch.float32)
            if X.ndim == 1:  # allow single-node input as (d,)
                X = X.unsqueeze(0)
            w = self.forward(X)  # (N,)
            return w.cpu().numpy()

    # Backward compat alias
    def predict(self, X_np):
        return self.predict_weights(X_np)
