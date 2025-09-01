import os, glob, csv


def collect(root):
    rows = {}
    for p in glob.glob(os.path.join(root, "*", "summary.csv")):
        exp = os.path.basename(os.path.dirname(p))
        try:
            with open(p) as f:
                r = list(csv.DictReader(f))
            if r:
                rows[exp] = {k: float(r[0][k]) for k in ["rmse_x", "rmse_y", "rmse_z", "rmse_3d", "nees"]}
        except Exception:
            pass
    return rows


def main():
    A = collect("outputs_baseline_v1")
    B1 = collect("outputs_online_tuned_v1")
    B2 = collect("outputs_online_tuned_v2")

    header = (
        f"{'exp':40s}  "
        f"{'rmse3d(A)':>10} {'rmse3d(B1)':>11} {'rmse3d(B2)':>11}  "
        f"{'ΔB1-A':>8} {'ΔB2-A':>8}   "
        f"{'nees(A)':>8} {'nees(B1)':>9} {'nees(B2)':>9}  "
        f"{'ΔB1-A':>8} {'ΔB2-A':>8}"
    )
    print(header)

    # Prefer rows common to A and any tuned set; start with those in A∩B2, then A∩B1, then others
    keys = sorted(set(A.keys()) & (set(B2.keys()) | set(B1.keys())))

    # Prepare CSV export
    out_dir = os.path.join("results")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "comparison_A_B1_B2.csv")
    import csv as _csv
    with open(out_csv, "w", newline="") as f:
        w = _csv.writer(f)
        w.writerow([
            "exp",
            "rmse3d_A", "rmse3d_B1", "rmse3d_B2",
            "d_rmse_B1_A", "d_rmse_B2_A",
            "nees_A", "nees_B1", "nees_B2",
            "d_nees_B1_A", "d_nees_B2_A",
        ])
        for k in keys:
            a = A.get(k)
            b1 = B1.get(k)
            b2 = B2.get(k)
            if not a:
                continue
            def fmt(v, w=10):
                return f"{v:>{w}.3f}" if v is not None else f"{'-':>{w}}"

            rm_a = a["rmse_3d"]
            ne_a = a["nees"]
            rm_b1 = b1["rmse_3d"] if b1 else None
            ne_b1 = b1["nees"] if b1 else None
            rm_b2 = b2["rmse_3d"] if b2 else None
            ne_b2 = b2["nees"] if b2 else None
            d1 = (rm_b1 - rm_a) if rm_b1 is not None else None
            d2 = (rm_b2 - rm_a) if rm_b2 is not None else None
            dn1 = (ne_b1 - ne_a) if ne_b1 is not None else None
            dn2 = (ne_b2 - ne_a) if ne_b2 is not None else None

            line = (
                f"{k:40s}  "
                f"{fmt(rm_a,10)} {fmt(rm_b1,11)} {fmt(rm_b2,11)}  "
                f"{fmt(d1,8)} {fmt(d2,8)}   "
                f"{fmt(ne_a,8)} {fmt(ne_b1,9)} {fmt(ne_b2,9)}  "
                f"{fmt(dn1,8)} {fmt(dn2,8)}"
            )
            print(line)
            # write CSV row (None -> empty)
            def nv(x):
                return ("" if x is None else f"{x:.6f}")
            w.writerow([
                k,
                f"{rm_a:.6f}", nv(rm_b1), nv(rm_b2), nv(d1), nv(d2),
                f"{ne_a:.6f}", nv(ne_b1), nv(ne_b2), nv(dn1), nv(dn2)
            ])

    print(f"\n[EXPORT] Wrote CSV: {out_csv}")


if __name__ == "__main__":
    main()
