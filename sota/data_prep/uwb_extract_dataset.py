"""
Builds an ML‑ready dataset for
    • NLOS/LOS classification
    • range‑bias regression
from MILUV UWB logs               ©2025 Your‑Thesis‑Team
"""
import argparse, pathlib, numpy as np, pandas as pd
from miluv.data import DataLoader


def merge_cir_and_range(cir_df, rng_df, max_dt=0.05):
    """
    Returns dataframe with columns:
        timestamp, from_id, to_id, cir, range, gt_range
    """
    # keep only columns we care about
    cir = cir_df[["timestamp", "from_id", "to_id", "cir"]].copy()
    rng = rng_df[["timestamp", "from_id", "to_id", "range", "gt_range"]].copy()

    # we need them sorted for merge_asof
    cir = cir.sort_values("timestamp")
    rng = rng.sort_values("timestamp")

    # merge within each (from,to) pair independently
    merged_chunks = []
    for (f, t), cir_pair in cir.groupby(["from_id", "to_id"]):
        if (f, t) not in rng.groupby(["from_id", "to_id"]).groups:
            continue
        rng_pair = rng[(rng["from_id"] == f) & (rng["to_id"] == t)]
        m = pd.merge_asof(
            rng_pair,
            cir_pair,
            on="timestamp",
            by=["from_id", "to_id"],
            direction="nearest",
            tolerance=max_dt,
        )
        m = m.dropna(subset=["cir"])  # rows without a nearby CIR
        merged_chunks.append(m)

    if not merged_chunks:
        return pd.DataFrame()

    merged = pd.concat(merged_chunks, ignore_index=True)
    return merged.reset_index(drop=True)


def build(exp_name: str, exp_root: str | None = None, max_dt: float = 0.05):
    dl = DataLoader(
        exp_name,
        exp_dir=exp_root,  # None ⇒ default
        cir=True,
        barometer=False,
        height=False,
    )

    samples = []
    for robot_id, robot in dl.data.items():
        cir_df = robot["uwb_cir"]
        rng_df = robot["uwb_range"]

        merged = merge_cir_and_range(cir_df, rng_df, max_dt=max_dt)
        if merged.empty:
            print(f"[warn] 0 matches in robot {robot_id}")
            continue

        merged["bias"] = merged["range"] - merged["gt_range"]
        merged["nlos"] = (merged["bias"].abs() > 0.15).astype(np.int8)
        samples.append(merged[["cir", "bias", "nlos"]])

        print(
            f"Robot {robot_id}: kept {len(merged)} / "
            f"{len(rng_df)} range packets (≈ {len(merged)/len(rng_df):.1%})"
        )

    if not samples:
        raise RuntimeError("No CIR–range pairs found – check timestamps or max_dt.")

    full = pd.concat(samples, ignore_index=True)

    # build numpy arrays ----------------------------------------------------
    X = np.stack(full["cir"].apply(lambda s: np.asarray(eval(s), dtype=np.float32)[:128]))
    y_bias = full["bias"].values.astype(np.float32)
    y_cls = full["nlos"].values.astype(np.int8)

    # train/val/test split 70‑15‑15 stratified on LOS/NLOS
    from sklearn.model_selection import train_test_split

    X_train, X_tmp, y_cls_train, y_cls_tmp, y_bias_train, y_bias_tmp = train_test_split(
        X, y_cls, y_bias, test_size=0.30, stratify=y_cls, random_state=0
    )
    X_val, X_test, y_cls_val, y_cls_test, y_bias_val, y_bias_test = train_test_split(
        X_tmp, y_cls_tmp, y_bias_tmp, test_size=0.50, stratify=y_cls_tmp, random_state=1
    )

    out = pathlib.Path("sota/uwb/datasets")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / f"{exp_name}.npz",
        X_train=X_train,
        y_cls_train=y_cls_train,
        y_reg_train=y_bias_train,
        X_val=X_val,
        y_cls_val=y_cls_val,
        y_reg_val=y_bias_val,
        X_test=X_test,
        y_cls_test=y_cls_test,
        y_reg_test=y_bias_test,
    )
    print("Saved →", out / f"{exp_name}.npz")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("exp_name")
    p.add_argument("--exp_root", default=None, help="override default data dir")
    p.add_argument("--max_dt", type=float, default=0.05,
                   help="maximum |Δt| [s] between CIR and range")
    args = p.parse_args()
    build(args.exp_name, args.exp_root, args.max_dt)
