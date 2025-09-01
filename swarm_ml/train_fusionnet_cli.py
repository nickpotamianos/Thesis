# swarm_ml/train_fusionnet_cli.py
import argparse, json, os, gzip
import numpy as np
from typing import List, Dict, Any
from .train_fusionnet import train_fusionnet

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
            X = np.asarray(obj["X"], dtype=float)
            mus = np.asarray(obj["mus"], dtype=float)
            Ps = np.asarray(obj["Ps"], dtype=float)
            gt = np.asarray(obj["gt_pos"], dtype=float)
            parts = {rid: (mus[i].copy(), Ps[i].copy()) for i, rid in enumerate(obj["order"]) }
            snaps.append({"X": X, "parts": parts, "gt_pos": gt})
    return snaps

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--snaps", required=True, help="Path to fusion_snaps.jsonl or .jsonl.gz")
    p.add_argument("--out", required=True, help="Output dir for model")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    snaps = load_fusion_snaps_jsonl(args.snaps)
    print(f"[CLI] Loaded {len(snaps):,} fusion snaps from {os.path.abspath(args.snaps)}")
    assert len(snaps) > 0, "No fusion snaps found (file exists but contained zero lines)."

    in_dim = int(snaps[0]["X"].shape[1])
    model = train_fusionnet(
        snaps=snaps,
        in_dim=in_dim,
        out_dir=args.out,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )
    print(f"[OK] FusionNet trained. Artifacts in: {args.out}")
