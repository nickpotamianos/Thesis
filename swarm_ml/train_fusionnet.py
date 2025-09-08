# swarm_ml/train_fusionnet.py
import argparse, os, json, random
import numpy as np, torch
from torch.utils.data import Dataset, DataLoader
from .models import FusionNet
import math

def _ci_fuse_torch(parts, w):
    """
    Torch implementation of CI fusion for differentiable training.

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
    # Final fusion
    P_fused = torch.inverse(J_sum)
    mu_fused = P_fused @ h_sum
    return mu_fused, P_fused

class FusionSnapDataset(Dataset):
    """
    PyTorch dataset for fusion snapshots.
    Each snapshot contains features X, parts dict, and gt_pos.
    """
    def __init__(self, snaps):
        self.snaps = snaps
    def __len__(self): return len(self.snaps)
    def __getitem__(self, i): return self.snaps[i]

def _stack_all_X(snaps, in_dim):
    Xs = []
    for s in snaps:
        X = np.asarray(s["X"], dtype=float)
        X = X.reshape(-1, in_dim)
        Xs.append(X)
    return np.vstack(Xs) if Xs else np.zeros((0, in_dim), dtype=float)

def _pearson(a: torch.Tensor, b: torch.Tensor) -> float:
    """Lightweight Pearson r without SciPy; safe for tiny variance."""
    a = a.detach().float(); b = b.detach().float()
    a = a - a.mean(); b = b - b.mean()
    denom = (a.std(unbiased=False) * b.std(unbiased=False)).item()
    if not math.isfinite(denom) or denom <= 1e-12:
        return float('nan')
    return float((a * b).mean().item() / denom)

def _weight_metrics(w: torch.Tensor, X: torch.Tensor | None) -> dict:
    """
    Return entropy (normalized to log(N)), HHI (∑w²), w_max, and
    correlations with a few canonical features if present:
      - reliability (idx=1), nis_ema (idx=7), var_pos_trace (idx=0)
    """
    N = max(1, int(w.shape[0]))
    ent = float((-(w * torch.log(w + 1e-9)).sum()).item())
    ent_norm = float(ent / max(1.0, math.log(N)))                 # ∈[0,1] where 1=uniform
    hhi = float((w * w).sum().item())                             # ∈[1/N,1], larger=w concentrated
    wmax = float(w.max().item())
    out = {"ent_norm": ent_norm, "hhi": hhi, "w_max": wmax}
    if X is not None and X.ndim == 2:
        try:
            if X.shape[1] > 1:  out["corr_rel"]     = _pearson(w, X[:, 1])   # reliability
            if X.shape[1] > 7:  out["corr_nis"]     = _pearson(w, X[:, 7])   # nis_ema
            if X.shape[1] > 0:  out["corr_varpos"]  = _pearson(w, X[:, 0])   # var_pos_trace
        except Exception:
            pass
    return out

def _epoch_nll(model, loader):
    """
    Validation pass with extra diagnostics:
      returns (avg_nll, stats_dict)
    """
    model.eval()
    nll_tot = 0.0; n = 0
    # accumulators for diagnostics
    acc = {"ent_norm": 0.0, "hhi": 0.0, "w_max": 0.0,
           "maha": 0.0, "logdet": 0.0,
           "corr_rel": 0.0, "corr_nis": 0.0, "corr_varpos": 0.0,
           "cnt": 0, "corr_cnt": 0}
    with torch.no_grad():
        for batch in loader:
            for snap in batch:
                X = torch.tensor(snap["X"], dtype=torch.float32)      # (N,d)
                y = torch.tensor(snap["gt_pos"], dtype=torch.float32) # (3,)
                w = model(X)                                          # (N,)
                mu_fused, P_fused = _ci_fuse_torch(snap["parts"], w)
                e = mu_fused[:3] - y
                P3 = P_fused[:3, :3] + torch.eye(3, dtype=torch.float32) * 1e-6
                maha = (e.unsqueeze(0) @ torch.linalg.solve(P3, e.unsqueeze(1))).squeeze()
                _, logdet = torch.slogdet(P3)
                nll = float(maha + logdet)
                nll_tot += nll; n += 1
                # weight stats
                m = _weight_metrics(w, X)
                acc["ent_norm"] += m["ent_norm"]; acc["hhi"] += m["hhi"]; acc["w_max"] += m["w_max"]
                acc["maha"] += float(maha); acc["logdet"] += float(logdet)
                acc["cnt"] += 1
                # correlations (may be NaN if feature missing)
                for k in ("corr_rel","corr_nis","corr_varpos"):
                    if k in m and math.isfinite(m[k]):
                        acc[k] += m[k]; acc["corr_cnt"] += 1
    stats = {
        "w_ent_norm": acc["ent_norm"] / max(1, acc["cnt"]),
        "w_hhi":      acc["hhi"]      / max(1, acc["cnt"]),
        "w_max":      acc["w_max"]    / max(1, acc["cnt"]),
        "maha":       acc["maha"]     / max(1, acc["cnt"]),
        "logdet":     acc["logdet"]   / max(1, acc["cnt"]),
    }
    if acc["corr_cnt"] > 0:
        stats["corr_rel"]     = acc["corr_rel"]    / acc["corr_cnt"]
        stats["corr_nis"]     = acc["corr_nis"]    / acc["corr_cnt"]
        stats["corr_varpos"]  = acc["corr_varpos"] / acc["corr_cnt"]
    return (nll_tot / max(n, 1)), stats

def train_fusionnet(
    train_snaps,
    val_snaps,
    in_dim,
    out_dir,
    epochs=50,
    lr=3e-4,
    batch_size=512,
    seed=42,
    patience=5,
    resume_dir=None,
):
    def _grad_global_norm(m: torch.nn.Module) -> float:
        tot = 0.0
        for p in m.parameters():
            if p.grad is not None:
                g = p.grad.detach().float()
                tot += float(torch.sum(g*g).item())
        return float(math.sqrt(max(0.0, tot)))

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    os.makedirs(out_dir, exist_ok=True)

    model = FusionNet(in_dim)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=False)

    # normalization from train set
    X_train = _stack_all_X(train_snaps, in_dim)
    if X_train.shape[0] > 0:
        mu = X_train.mean(axis=0)
        sd = X_train.std(axis=0)
        sd[sd < 1e-6] = 1.0
        model.set_normalizer(mu, sd)
        print(f"[norm] feature_mean[min,max]=({mu.min():.3f},{mu.max():.3f}) "
              f"feature_std[min,max]=({sd.min():.3f},{sd.max():.3f})")

    # --- Resume (optional) ---
    if resume_dir is not None:
        try:
            state = torch.load(os.path.join(resume_dir, "fusionnet.pt"), map_location="cpu")
            model.load_state_dict(state, strict=False)
            # Load normalizer if available
            meta_p = os.path.join(resume_dir, "fusionnet_meta.json")
            if os.path.exists(meta_p):
                with open(meta_p, "r") as f:
                    meta = json.load(f)
                if "x_mu" in meta and "x_std" in meta:
                    model.set_normalizer(meta["x_mu"], meta["x_std"])
            print(f"[resume] Loaded weights from {resume_dir}")
        except Exception as e:
            print(f"[resume] Warning: could not resume from {resume_dir}: {e}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, verbose=False)

    tr_ds = FusionSnapDataset(train_snaps)
    va_ds = FusionSnapDataset(val_snaps)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  collate_fn=lambda b: b)
    va_dl = DataLoader(va_ds, batch_size=batch_size, shuffle=False, collate_fn=lambda b: b)

    best_val = 1e9
    best_ep = -1
    bad = 0

    for ep in range(epochs):
        model.train()
        tot = 0.0; cnt = 0
        # epoch accumulators for diagnostics
        acc = {"ent_norm": 0.0, "hhi": 0.0, "w_max": 0.0,
               "maha": 0.0, "logdet": 0.0, "g_norm": 0.0,
               "corr_rel": 0.0, "corr_nis": 0.0, "corr_varpos": 0.0,
               "cnt": 0, "corr_cnt": 0}
        for batch in tr_dl:
            loss = 0.0
            for snap in batch:
                X = torch.tensor(snap["X"], dtype=torch.float32)      # (N,d)
                y = torch.tensor(snap["gt_pos"], dtype=torch.float32) # (3,)
                w = model(X)                                          # (N,)
                mu_fused, P_fused = _ci_fuse_torch(snap["parts"], w)
                e = mu_fused[:3] - y
                P3 = P_fused[:3, :3] + torch.eye(3, dtype=torch.float32) * 1e-6
                maha = (e.unsqueeze(0) @ torch.linalg.solve(P3, e.unsqueeze(1))).squeeze()
                _, logdet = torch.slogdet(P3)
                nll = maha + logdet
                entropy = -(w * torch.log(w + 1e-8)).sum()
                loss = loss + (nll - 0.01 * entropy)
                # accumulate diagnostics (per-snap)
                m = _weight_metrics(w, X)
                acc["ent_norm"] += m["ent_norm"]; acc["hhi"] += m["hhi"]; acc["w_max"] += m["w_max"]
                acc["maha"] += float(maha); acc["logdet"] += float(logdet); acc["cnt"] += 1
                for k in ("corr_rel","corr_nis","corr_varpos"):
                    if k in m and math.isfinite(m[k]):
                        acc[k] += m[k]; acc["corr_cnt"] += 1
            loss = loss / len(batch)
            opt.zero_grad()
            loss.backward()
            # gradient norm BEFORE clipping (health)
            acc["g_norm"] += _grad_global_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            tot += loss.item(); cnt += 1
        tr_avg = tot / max(cnt, 1)

        va_avg, va_stats = _epoch_nll(model, va_dl)
        sched.step(va_avg)
        # finalize train stats
        tr_stats = {
            "w_ent_norm": acc["ent_norm"] / max(1, acc["cnt"]),
            "w_hhi":      acc["hhi"]      / max(1, acc["cnt"]),
            "w_max":      acc["w_max"]    / max(1, acc["cnt"]),
            "maha":       acc["maha"]     / max(1, acc["cnt"]),
            "logdet":     acc["logdet"]   / max(1, acc["cnt"]),
            "g_norm":     acc["g_norm"]   / max(1, cnt),
        }
        if acc["corr_cnt"] > 0:
            tr_stats["corr_rel"]    = acc["corr_rel"]    / acc["corr_cnt"]
            tr_stats["corr_nis"]    = acc["corr_nis"]    / acc["corr_cnt"]
            tr_stats["corr_varpos"] = acc["corr_varpos"] / acc["corr_cnt"]

        # pretty print one compact, information-dense line
        lr_now = opt.param_groups[0]['lr']
        def g(d, k): return float(d.get(k, float('nan')))
        print(
            f"[ep {ep+1:03d}] "
            f"train_nll={tr_avg:.4f}  val_nll={va_avg:.4f}  lr={lr_now:.2e} | "
            f"w(ent_norm)={g(tr_stats,'w_ent_norm'):.2f}/{g(va_stats,'w_ent_norm'):.2f}  "
            f"w(HHI)={g(tr_stats,'w_hhi'):.3f}/{g(va_stats,'w_hhi'):.3f}  "
            f"w_max={g(tr_stats,'w_max'):.2f}/{g(va_stats,'w_max'):.2f} | "
            f"corr[w,rel]={g(tr_stats,'corr_rel'):.2f}/{g(va_stats,'corr_rel'):.2f}  "
            f"corr[w,nis]={g(tr_stats,'corr_nis'):.2f}/{g(va_stats,'corr_nis'):.2f}  "
            f"corr[w,var]={g(tr_stats,'corr_varpos'):.2f}/{g(va_stats,'corr_varpos'):.2f} | "
            f"maha={g(tr_stats,'maha'):.3f}/{g(va_stats,'maha'):.3f}  "
            f"logdet={g(tr_stats,'logdet'):.3f}/{g(va_stats,'logdet'):.3f}  "
            f"||grad||={g(tr_stats,'g_norm'):.2e}"
        )

        if va_avg + 1e-6 < best_val:
            best_val = va_avg; best_ep = ep; bad = 0
            torch.save(model.state_dict(), os.path.join(out_dir, "fusionnet.pt"))
            with open(os.path.join(out_dir, "fusionnet_meta.json"), "w") as f:
                json.dump({"in_dim": in_dim, "x_mu": model.x_mu.squeeze(0).tolist(), "x_std": model.x_std.squeeze(0).tolist()}, f, indent=2)
        else:
            bad += 1
            if bad >= patience:
                print(f"[early-stop] no val improvement for {patience} epochs (best ep {best_ep+1})")
                break

    # Save final meta even if no improvement (keeps normalizer available)
    if not os.path.exists(os.path.join(out_dir, "fusionnet_meta.json")):
        with open(os.path.join(out_dir, "fusionnet_meta.json"), "w") as f:
            json.dump({"in_dim": in_dim, "x_mu": model.x_mu.squeeze(0).tolist(), "x_std": model.x_std.squeeze(0).tolist()}, f, indent=2)
    return model