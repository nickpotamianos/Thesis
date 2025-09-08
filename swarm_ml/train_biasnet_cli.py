# swarm_ml/train_biasnet_cli.py
import argparse, os, json, random
from typing import List, Dict
from .datasets import load_bias_samples_jsonl
from .train_biasnet import train_biasnet

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="Path to bias_samples.jsonl or .jsonl.gz")
    p.add_argument("--out", required=True, help="Output dir for model")
    p.add_argument("--val_split", type=float, default=None, help="Alias for --val_ratio")
    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--split_mode", choices=["random","by_time","by_exp","by_tracker"], default="random")
    p.add_argument("--val_exps", type=str, default=None, help="Comma-separated experiment ids to hold out when --split_mode=by_exp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=256)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    samples: List[Dict] = load_bias_samples_jsonl(args.samples)
    print(f"[CLI] Loaded {len(samples):,} bias samples from {os.path.abspath(args.samples)}")
    random.seed(args.seed)

    # ----- split helpers -----
    def _by_time_split(ss):
        ss_sorted = sorted(ss, key=lambda s: float(s.get("meta", {}).get("timestamp", 0.0)))
        n_total = len(ss_sorted)
        n_val = max(1, int((args.val_split if args.val_split is not None else args.val_ratio) * n_total))
        return ss_sorted[:-n_val], ss_sorted[-n_val:]

    def _by_exp_split(ss):
        val_exps = set([s.strip() for s in args.val_exps.split(",")]) if args.val_exps else set()
        if not val_exps:
            # fallback: hash split by exp
            exps = sorted(set(s.get("meta", {}).get("exp", "") for s in ss))
            n_val_e = max(1, int(round(len(exps) * (args.val_split if args.val_split is not None else args.val_ratio))))
            random.shuffle(exps)
            val_exps = set(exps[:n_val_e])
        val = [s for s in ss if s.get("meta", {}).get("exp", None) in val_exps]
        tr  = [s for s in ss if s.get("meta", {}).get("exp", None) not in val_exps]
        return tr, val

    def _by_tracker_split(ss):
        trks = sorted(set(s.get("meta", {}).get("tracker","") for s in ss))
        n_val_t = max(1, int(round(len(trks) * (args.val_split if args.val_split is not None else args.val_ratio))))
        random.shuffle(trks)
        val_trks = set(trks[:n_val_t])
        val = [s for s in ss if s.get("meta", {}).get("tracker","") in val_trks]
        tr  = [s for s in ss if s.get("meta", {}).get("tracker","") not in val_trks]
        return tr, val

    # ----- choose split -----
    if args.split_mode == "by_time":
        train_samples, val_samples = _by_time_split(samples)
    elif args.split_mode == "by_exp":
        train_samples, val_samples = _by_exp_split(samples)
    elif args.split_mode == "by_tracker":
        train_samples, val_samples = _by_tracker_split(samples)
    else:
        # random split (reproducible)
        samples_shuf = samples[:]
        random.shuffle(samples_shuf)
        n_total = len(samples_shuf)
        n_val = max(1, int((args.val_split if args.val_split is not None else args.val_ratio) * n_total))
        val_samples = samples_shuf[:n_val]
        train_samples = samples_shuf[n_val:]

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
