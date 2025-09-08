# eval_baselines.py
import json, gzip, os, numpy as np
from math import isnan, isfinite
from swarm_ml.fusion import CIFuser, CIFuserConfig

def _open_any(p):
    if os.path.exists(p): return gzip.open(p,"rt") if p.endswith(".gz") else open(p,"r")
    if p.endswith(".gz") and os.path.exists(p[:-3]): return open(p[:-3],"r")
    if p.endswith(".jsonl") and os.path.exists(p+".gz"): return gzip.open(p+".gz","rt")
    raise FileNotFoundError(p)

def load_snaps(path):
    snaps=[]
    with _open_any(path) as f:
        for line in f:
            o=json.loads(line); X=np.asarray(o["X"],float)
            mus=np.asarray(o["mus"],float); Ps=np.asarray(o["Ps"],float)
            parts={rid:(mus[i],Ps[i]) for i,rid in enumerate(o["order"])}
            snaps.append({"X":X,"parts":parts,"gt_pos":np.asarray(o["gt_pos"],float),
                          "exp":o.get("exp")})
    return snaps

def nll_of(mu,P,y):
    e=mu[:3]-y; P3=P[:3,:3]
    # jitter for numerical stability
    w,v=np.linalg.eigh(P3); w=np.clip(w,1e-9,None); P3 = (v*w)@v.T
    maha = float(e.T @ np.linalg.solve(P3,e))
    sign,logdet=np.linalg.slogdet(P3)
    return float(maha + logdet)

snaps = load_snaps("data/fusion_snaps_all.jsonl")
val = [s for s in snaps if s["exp"]=="default_3_random3_2"]
train = [s for s in snaps if s["exp"]!="default_3_random3_2"]
print(f"val snaps: {len(val)}, train snaps: {len(train)}")

fuser_grid = CIFuser(CIFuserConfig(objective="logdet", grid_step=0.1))
def fuse(parts, method):
    if method=="uniform":
        keys=list(parts.keys()); w={k:1.0/len(keys) for k in keys}
        from numpy.linalg import inv
        J=h=None
        for k,(mu,P) in parts.items():
            Ji = np.linalg.inv(P + 1e-6*np.eye(P.shape[0])); hi=Ji@mu
            if J is None: J, h = Ji, hi
            else: J, h = J+Ji, h+hi
        P = np.linalg.inv(J); mu=P@h
        return mu,P
    else:
        mu,P,_ = fuser_grid.fuse(parts, method="grid")
        return mu,P

print("\n=== Baseline NLL Analysis ===")
for method in ["uniform","grid"]:
    # Compute NLL on training set
    train_nll=0.0; train_cnt=0
    for s in train:
        mu,P = fuse(s["parts"], method)
        train_nll += nll_of(mu,P,s["gt_pos"]); train_cnt+=1
    train_avg = train_nll/max(train_cnt,1)
    
    # Compute NLL on validation set
    val_nll=0.0; val_cnt=0
    for s in val:
        mu,P = fuse(s["parts"], method)
        val_nll += nll_of(mu,P,s["gt_pos"]); val_cnt+=1
    val_avg = val_nll/max(val_cnt,1)
    
    print(f"{method:8s} | Train NLL: {train_avg:6.2f} | Val NLL: {val_avg:6.2f} | Gap: {val_avg/train_avg:.1f}x")

# Check node count distribution
print("\n=== Node Count Analysis ===")
train_nodes = [len(s["parts"]) for s in train]
val_nodes = [len(s["parts"]) for s in val]
print(f"Train nodes - mean: {np.mean(train_nodes):.2f}, std: {np.std(train_nodes):.2f}")
print(f"Val nodes   - mean: {np.mean(val_nodes):.2f}, std: {np.std(val_nodes):.2f}")

# Feature variance analysis
print("\n=== Feature Variance Analysis ===")
feature_names = ['var_pos', 'reliability', 'z_agg', 'R_eff', 'geom_ez', 'los_score', 'gate_sigma', 'nis_ema']
train_X = np.vstack([s["X"] for s in train])
val_X = np.vstack([s["X"] for s in val])

print("Feature      | Train Std | Val Std   | Ratio")
print("-------------|-----------|-----------|--------")
for i, fname in enumerate(feature_names):
    if i < train_X.shape[1]:
        train_std = np.std(train_X[:, i])
        val_std = np.std(val_X[:, i])
        ratio = val_std / max(train_std, 1e-8)
        status = "Static" if train_std < 1e-6 else "Dynamic"
        print(f"{fname:12s} | {train_std:9.6f} | {val_std:9.6f} | {ratio:6.2f} | {status}")