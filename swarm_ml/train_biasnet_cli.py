import argparse, os, json, random
from typing import List, Dict
from .datasets import load_bias_samples_jsonl, BiasNetDataset
from .train_biasnet import train_biasnet

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="Path to bias_samples.jsonl")
    p.add_argument("--out", required=True, help="Output dir for model")
    # Support both --val_split and --val_ratio for convenience
    p.add_argument("--val_split", type=float, default=None)
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    random.seed(args.seed)

    samples: List[Dict] = load_bias_samples_jsonl(args.samples)
    # Shuffle and split
    random.shuffle(samples)
    n_total = len(samples)
    val_ratio = args.val_ratio if (args.val_split is None) else args.val_split
    n_val = max(1, int(val_ratio * n_total))
    val_samples = samples[:n_val]
    train_samples = samples[n_val:]

    assert len(train_samples) > 0 and len(val_samples) > 0, "Not enough samples after split"

    in_dim = len(train_samples[0]["features"])
    model = train_biasnet(
        train_samples=train_samples,
        val_samples=val_samples,
        in_dim=in_dim,
        out_dir=args.out,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        seed=args.seed
    )

    print(f"[OK] BiasNet trained. Artifacts in: {args.out}")
