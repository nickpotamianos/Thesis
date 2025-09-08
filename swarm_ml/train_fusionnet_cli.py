# swarm_ml/train_fusionnet_cli.py
import argparse, json, os, gzip, random
import numpy as np
import math
from typing import List, Dict, Any
from .train_fusionnet import train_fusionnet

import numpy as np
def _open_any(path: str):
    """
    Open either plain JSONL or GZ; auto-fallback across .jsonl <-> .jsonl.gz
    """
    if os.path.exists(path):
        return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")
    if path.endswith(".gz") and os.path.exists(path[:-3]):
        return open(path[:-3], "r")
    if path.endswith(".jsonl") and os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rt")
    raise FileNotFoundError(f"Fusion snaps file not found. Tried: {path}, {path[:-3] if path.endswith('.gz') else ''}, {path + '.gz' if path.endswith('.jsonl') else ''}")

def load_fusion_snaps_jsonl(path: str):
    snaps = []
    with _open_any(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Optional meta for split-aware training
            ts = float(obj.get("timestamp", np.nan)) if "timestamp" in obj else np.nan
            exp = obj.get("exp", None)
            X = np.asarray(obj["X"], dtype=float)
            mus = np.asarray(obj["mus"], dtype=float)
            Ps = np.asarray(obj["Ps"], dtype=float)
            gt = np.asarray(obj["gt_pos"], dtype=float)
            parts = {rid: (mus[i].copy(), Ps[i].copy()) for i, rid in enumerate(obj["order"]) }
            snaps.append({"X": X, "parts": parts, "gt_pos": gt, "timestamp": ts, "exp": exp})
    return snaps

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snaps", required=True, help="Path to fusion_snaps.jsonl or .jsonl.gz")
    p.add_argument("--out", required=True, help="Output dir for model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--patience", type=int, default=5, help="Early-stop patience on val_nll")
    p.add_argument("--resume_from", type=str, default=None, help="Optional directory with fusionnet.pt/meta to resume from")
    p.add_argument("--split_mode", choices=["random","by_time","by_exp"], default="random")
    p.add_argument("--val_exps", type=str, default=None, help="Comma-separated experiment ids to hold out when --split_mode=by_exp")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    snaps = load_fusion_snaps_jsonl(args.snaps)
    print(f"[CLI] Loaded {len(snaps):,} fusion snaps from {os.path.abspath(args.snaps)}")
    assert len(snaps) > 0, "No fusion snaps found (file exists but contained zero lines)."
    try:
        n_nodes = [int(np.asarray(s['X']).shape[0]) for s in snaps]
        print(f"[CLI] nodes/snap: mean={np.mean(n_nodes):.1f}  median={np.median(n_nodes):.0f}  "
              f"min={np.min(n_nodes)}  max={np.max(n_nodes)}")
    except Exception:
        pass

    random.seed(args.seed)

    # Split helpers
    def _split_random(ss):
        ss2 = ss[:]; random.shuffle(ss2)
        n_val = max(1, int(0.2 * len(ss2)))
        return ss2[n_val:], ss2[:n_val]
    def _split_time(ss):
        ss2 = sorted(ss, key=lambda s: float(s.get("timestamp", np.nan)))
        n_val = max(1, int(0.2 * len(ss2)))
        return ss2[:-n_val], ss2[-n_val:]
    def _split_exp(ss):
        val_exps = set([s.strip() for s in args.val_exps.split(",")]) if args.val_exps else set()
        if not val_exps:
            exps = sorted(set(s.get("exp", None) for s in ss))
            random.shuffle(exps)
            val_exps = set(exps[:max(1, int(round(0.2*len(exps))))])
        tr = [s for s in ss if s.get("exp", None) not in val_exps]
        va = [s for s in ss if s.get("exp", None) in val_exps]
        return tr, va

    if args.split_mode == "by_time":
        train_snaps, val_snaps = _split_time(snaps)
    elif args.split_mode == "by_exp":
        train_snaps, val_snaps = _split_exp(snaps)
    else:
        train_snaps, val_snaps = _split_random(snaps)

    in_dim = int(train_snaps[0]["X"].shape[1])
    print(f"[CLI] split: train={len(train_snaps)}  val={len(val_snaps)}  in_dim={in_dim}")
    model = train_fusionnet(
        train_snaps=train_snaps,
        val_snaps=val_snaps,
        in_dim=in_dim,
        out_dir=args.out,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience,
        resume_dir=args.resume_from
    )
    print(f"[OK] FusionNet trained. Artifacts in: {os.path.abspath(args.out)}")
