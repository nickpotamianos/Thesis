# swarm_ml/train_biasnet.py
import argparse, os, json
import numpy as np, torch
from torch.utils.data import DataLoader
from .models import BiasNet
from .datasets import BiasNetDataset

def train_biasnet(train_samples, val_samples, in_dim, out_dir, lr=1e-3, epochs=30, batch_size=256, seed=0):
    torch.manual_seed(seed)
    model = BiasNet(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = torch.nn.SmoothL1Loss()

    train_ds = BiasNetDataset(train_samples)
    val_ds = BiasNetDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val = 1e9
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            opt.zero_grad()
            yhat = model(xb)
            loss = loss_fn(yhat, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                yhat = model(xb)
                va_loss += loss_fn(yhat, yb).item() * xb.size(0)
        va_loss /= len(val_ds)

        print(f"[ep {ep+1:03d}] train={tr_loss:.4f} val={va_loss:.4f}")
        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "biasnet.pt"))

    with open(os.path.join(out_dir, "biasnet_meta.json"), "w") as f:
        json.dump({"in_dim": in_dim}, f, indent=2)
    return model
