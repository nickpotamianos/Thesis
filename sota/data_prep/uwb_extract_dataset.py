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

    # add symmetric copy of rng so that (from,to) or (to,from) both match
    rng_s = rng.rename(columns={"from_id": "to_id", "to_id": "from_id"})
    rng_all = pd.concat([rng, rng_s], ignore_index=True).sort_values("timestamp")

    # delay correction
    import pickle, pathlib as _pl
    _delay = pickle.load(open("config/uwb/uwb_calib.pickle","rb"))["delays"]
    def _delay_correct(f_id, t_id, r):
        return r - (_delay[f_id] + _delay[t_id])

    pairs = cir.groupby(["from_id", "to_id"])
    merged_chunks = []
    for (f, t), cir_pair in pairs:
        sub_rng = rng_all[(rng_all["from_id"] == f) & (rng_all["to_id"] == t)]
        if sub_rng.empty:
            continue
        m = pd.merge_asof(
            cir_pair.sort_values("timestamp"),
            sub_rng,
            on="timestamp",
            by=["from_id", "to_id"],
            direction="nearest",
            tolerance=max_dt,
        ).dropna(subset=["range"])
        
        if not m.empty:
            # apply delay correction
            m["range_dc"] = m.apply(
                lambda r: _delay_correct(r["from_id"], r["to_id"], r["range"]),
                axis=1)
        merged_chunks.append(m)

    return pd.concat(merged_chunks, ignore_index=True) if merged_chunks else pd.DataFrame()


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

        # adaptive tolerance to handle clock offsets
        merged = None
        for tol in (0.05, 0.10, 0.20, 0.40):          # seconds
            merged = merge_cir_and_range(cir_df, rng_df, max_dt=tol)
            if len(merged) / len(rng_df) > 0.60:       # keep ≥60 % ?
                print(f"  ✓ tolerance {tol:0.2f}s gives {len(merged)} matches")
                break
            print(f"    tolerance {tol:0.2f}s only {len(merged)} matches")
        else:
            print(f"[warn] even 0.4 s tolerance gives few matches → keeping anyway")

        if merged.empty:
            print(f"[warn] 0 matches in robot {robot_id}")
            continue

        # use delay-corrected range for bias calculation
        merged["bias"] = merged["range_dc"] - merged["gt_range"]
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
    y_bias = np.clip(full["bias"].values.astype(np.float32), -0.5, 0.5)
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
