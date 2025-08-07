"""
Builds an ML‑ready dataset for
    • NLOS/LOS classification
    • range‑bias regression
from MILUV UWB logs               ©2025 Your‑Thesis‑Team
"""
import argparse, pathlib, numpy as np, pandas as pd
from miluv.data import DataLoader


def merge_cir_and_range(cir_df: pd.DataFrame,
                        rng_df: pd.DataFrame,
                        max_dt: float = 0.12) -> pd.DataFrame:
    """
    Return a table with aligned (CIR, range, gt_range).
    Handles both      (from,to)  and  (to,from)
    and allows |Δt| ≤ max_dt  (default 120 ms).
    """
    cir = cir_df[["timestamp", "from_id", "to_id", "cir"]].copy()
    rng = rng_df[["timestamp", "from_id", "to_id", "range", "gt_range"]].copy()

    # ------------------------------------------------------------------ #
    # 1) make every table direction‑agnostic by adding a swapped copy
    def _swap(df, extra_cols=None):
        cols = ["timestamp", "from_id", "to_id"] + (extra_cols or [])
        out = df[cols].copy()
        out[["from_id", "to_id"]] = out[["to_id", "from_id"]]
        return out

    cir_all = pd.concat([cir, _swap(cir, ["cir"])], ignore_index=True)
    rng_all = pd.concat([rng, _swap(rng, ["range", "gt_range"])], ignore_index=True)

    cir_all = cir_all.sort_values("timestamp")
    rng_all = rng_all.sort_values("timestamp")

    # ------------------------------------------------------------------ #
    # 2) merge_asof within each *unordered* pair (min(id),max(id))
    cir_all["pair_key"] = list(zip(cir_all[["from_id", "to_id"]]
                                        .min(axis=1),
                                   cir_all[["from_id", "to_id"]]
                                        .max(axis=1)))
    rng_all["pair_key"] = list(zip(rng_all[["from_id", "to_id"]]
                                        .min(axis=1),
                                   rng_all[["from_id", "to_id"]]
                                        .max(axis=1)))

    merged = []
    for key, cir_grp in cir_all.groupby("pair_key"):
        rng_grp = rng_all[rng_all["pair_key"] == key]
        if rng_grp.empty:
            continue
        m = pd.merge_asof(cir_grp, rng_grp,
                          on="timestamp",
                          direction="nearest",
                          tolerance=max_dt,
                          suffixes=("_cir", "_rng"))
        m = m.dropna(subset=["range", "cir"])
        merged.append(m)

    return (pd.concat(merged, ignore_index=True)
            if merged else pd.DataFrame())


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

        # delay correction
        import pickle
        _delay = pickle.load(open("config/uwb/uwb_calib.pickle","rb"))["delays"]
        def _delay_correct(f_id, t_id, r):
            return r - (_delay[f_id] + _delay[t_id])

        # apply delay correction - use _rng suffixed columns from merge
        merged["range_dc"] = merged.apply(
            lambda row: _delay_correct(row["from_id_rng"], row["to_id_rng"], row["range"]),
            axis=1)

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
