import argparse, json, os, gzip
import numpy as np
from typing import List, Dict, Any
from .train_fusionnet import train_fusionnet

def _iter_lines(path: str):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            for line in f:
                yield line
    else:
        with open(path, 'r') as f:
            for line in f:
                yield line

def load_fusion_snaps_jsonl(path: str):
    snaps = []
    for line in _iter_lines(path):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Convert lists to arrays
            X = np.asarray(obj["X"], dtype=float)
            mus = np.asarray(obj["mus"], dtype=float)
            Ps = np.asarray(obj["Ps"], dtype=float)
            gt = np.asarray(obj["gt_pos"], dtype=float)
            parts = {}
            for i, rid in enumerate(obj["order"]):
                parts[rid] = (mus[i].copy(), Ps[i].copy())
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
    assert len(snaps) > 0, "No fusion snaps found."

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
