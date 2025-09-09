import os, json, random
import numpy as np, torch
from torch.utils.data import DataLoader
from .models import BiasNet
from .datasets import BiasNetDataset

def train_biasnet(train_samples, val_samples, in_dim, out_dir, lr=1e-3, epochs=30, batch_size=256, seed=0):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    model = BiasNet(in_dim=in_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # install normalizer from train set
    try:
        X_train = np.asarray([s["features"] for s in train_samples], dtype=float)
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd < 1e-6] = 1.0
        model.set_normalizer(mu, sd)
        print(f"[BiasNet norm] feature_mean[min,max]=({mu.min():.3f},{mu.max():.3f}) feature_std[min,max]=({sd.min():.3f},{sd.max():.3f})")
    except Exception as e:
        print(f"[BiasNet norm] Warning: failed to compute/install normalizer: {e}")

    train_ds = BiasNetDataset(train_samples)
    val_ds = BiasNetDataset(val_samples)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    best_val = 1e9
    os.makedirs(out_dir, exist_ok=True)
    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        for batch in train_loader:
            x, y = batch["features"], batch["bias"]
            pred = model(x)
            loss = torch.nn.functional.mse_loss(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch["features"], batch["bias"]
                pred = model(x)
                loss = torch.nn.functional.mse_loss(pred, y)
                va_loss += loss.item()
        va_loss /= len(val_loader)

        print(f"[ep {ep+1:03d}] train={tr_loss:.4f} val={va_loss:.4f}")

        if va_loss < best_val:
            best_val = va_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "biasnet.pt"))

    # Save meta + normalizer for runtime parity
    meta = {"in_dim": in_dim}
    try:
        meta["x_mu"] = model.x_mu.squeeze(0).cpu().tolist()
        meta["x_std"] = model.x_std.squeeze(0).cpu().tolist()
    except Exception:
        pass
    with open(os.path.join(out_dir, "biasnet_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return model
