# swarm_ml/train_fusionnet.py
import argparse, os, json
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .models import FusionNet

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
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best = 1e9
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot = 0.0; cnt = 0
        for batch in dl:
            loss = 0.0
            for snap in batch:
                X = torch.tensor(snap["X"], dtype=torch.float32)      # (N,d)
                y = torch.tensor(snap["gt_pos"], dtype=torch.float32) # (3,)
                w = model(X)  # (N,)
                # Fuse using predicted weights (CI) outside torch (small N): detach to numpy
                w_np = w.detach().cpu().numpy()
                parts = snap["parts"]
                keys = list(parts.keys())
                weights = {k: float(w_np[i]) for i, k in enumerate(keys)}
                mu_fused, _ = fuser._fuse_given_weights(parts, weights)
                mu_t = torch.tensor(mu_fused[:3], dtype=torch.float32)  # position component

                loss = loss + torch.mean((mu_t - y)**2)
            loss = loss / len(batch)

            opt.zero_grad()
            loss.backward()
            opt.step()

            tot += loss.item(); cnt += 1

        avg = tot / max(cnt, 1)
        print(f"[ep {ep+1:03d}] train_mse={avg:.4f}")

        if avg < best:
            best = avg
            torch.save(model.state_dict(), os.path.join(out_dir, "fusionnet.pt"))
    with open(os.path.join(out_dir, "fusionnet_meta.json"), "w") as f:
        json.dump({"in_dim": in_dim}, f, indent=2)
    return model
