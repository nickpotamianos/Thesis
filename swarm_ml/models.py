# swarm_ml/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiasNet(nn.Module):
    """
    Small MLP that predicts scalar bias for a single range measurement.
    Input features from features.build_measurement_features(...)
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
    Given per-node features (d-dimensional), produce unnormalized scores s_i.
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
        H = self.enc(X)           # (N,h)
        s = self.score(H).squeeze(-1)  # (N,)
        w = F.softmax(s, dim=0)
        return w  # (N,)

    def predict(self, x_np):
        with torch.no_grad():
            X = torch.tensor(x_np, dtype=torch.float32)
            w = self.forward(X).cpu().numpy()
        # Return scalar if single feature, else average
        return float(w.mean()) if w.ndim == 1 else float(w.squeeze())
