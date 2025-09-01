# swarm_ml/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasNet(nn.Module):
    """
    Small MLP that predicts scalar bias for a single UWB range measurement.
    Input features must match features.build_measurement_features(...).
    """
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
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
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU()
        )
        self.score = nn.Linear(hidden, 1)

    def forward(self, X):  # X: (N_nodes, d)
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
