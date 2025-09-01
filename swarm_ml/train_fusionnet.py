# swarm_ml/train_fusionnet.py
import argparse, os, json
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .models import FusionNet

def _ci_fuse_torch(parts, w):
    """Torch implementation of CI fusion for differentiable training.
    parts: dict[id] -> (mu_i (6,), P_i (6,6)) numpy arrays
    w: torch.Tensor shape (N,) softmax weights
    Returns: (mu_fused (6,), P_fused (6,6)) torch tensors
    """
    keys = list(parts.keys())
    J_sum = None
    h_sum = None
    for i, rid in enumerate(keys):
        mu_i_np, P_i_np = parts[rid]
        mu_i = torch.tensor(mu_i_np, dtype=torch.float32)
        P_i = torch.tensor(P_i_np, dtype=torch.float32)
        # Numerical jitter for stability
        P_i = P_i + torch.eye(P_i.shape[0], dtype=torch.float32) * 1e-6
        J_i = torch.inverse(P_i)
        h_i = J_i @ mu_i
        wi = w[i]
        if J_sum is None:
            J_sum = wi * J_i
            h_sum = wi * h_i
        else:
            J_sum = J_sum + wi * J_i
            h_sum = h_sum + wi * h_i
    P = torch.inverse(J_sum)
    mu = P @ h_sum
    return mu, P

class FusionSnapDataset(Dataset):
    """
    Each sample contains:
      - X: (N_nodes, d) per-tracker node features (e.g., local NEES, LOS score, range residual stats)
      - y: target 3D position (ground truth) at that time
      - parts: list of (mu_i, P_i) local posteriors (not used directly in loss;
               loss defined on fused estimate from predicted weights)
    For training, we approximate fused estimate via weights -> CI fusion outside the dataset.
    """
    def __init__(self, snaps):
        self.snaps = snaps

    def __len__(self): return len(self.snaps)

    def __getitem__(self, i): return self.snaps[i]

def train_fusionnet(snaps, in_dim, out_dir, lr=1e-3, epochs=20, batch_size=64, seed=0, fuser=None):
    torch.manual_seed(seed)
    model = FusionNet(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    ds = FusionSnapDataset(snaps)
    # Keep batches as list[dict] to simplify processing
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=lambda b: b)

    best = 1e9
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0; cnt = 0
        for batch in dl:
            loss = 0.0
            # batch is a list of dict samples
            for snap in batch:
                X = torch.tensor(snap["X"], dtype=torch.float32)      # (N,d)
                y = torch.tensor(snap["gt_pos"], dtype=torch.float32) # (3,)
                w = model(X)  # (N,)
                parts = snap["parts"]
                mu_fused, P_fused = _ci_fuse_torch(parts, w)
                mu_t = mu_fused[:3]
                # Gaussian NLL: (e^T P^{-1} e) + logdet(P)
                e = (mu_t - y)
                P3 = P_fused[:3, :3] + torch.eye(3) * 1e-6
                # solve for Mahalanobis term
                maha = torch.matmul(e.unsqueeze(0), torch.linalg.solve(P3, e.unsqueeze(1))).squeeze()
                sign, logdet = torch.slogdet(P3)
                nll = maha + logdet
                # Entropy regularizer on weights (encourage non-degenerate)
                eps = 1e-8
                entropy = -(w * torch.log(w + eps)).sum()
                loss = loss + (nll - 0.01 * entropy)
            loss = loss / len(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item(); cnt += 1

        avg = tot / max(cnt, 1)
        print(f"[ep {ep+1:03d}] train_nll={avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out_dir, "fusionnet.pt"))
    with open(os.path.join(out_dir, "fusionnet_meta.json"), "w") as f:
        json.dump({"in_dim": in_dim}, f, indent=2)
    return model
