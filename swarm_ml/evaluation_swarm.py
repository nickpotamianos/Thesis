# swarm_ml/evaluation_swarm.py
import os, csv
import numpy as np

def rmse_xyz(est: np.ndarray, gt: np.ndarray) -> dict:
    assert est.shape == gt.shape
    e = est - gt
    mse = np.mean(e**2, axis=0)
    rmse = np.sqrt(mse)
    return {"rmse_x": float(rmse[0]), "rmse_y": float(rmse[1]), "rmse_z": float(rmse[2]),
            "rmse_3d": float(np.sqrt(np.mean(np.sum(e**2, axis=1))))}

def nees(traj_mu: np.ndarray, traj_P: np.ndarray, gt: np.ndarray) -> float:
    N = traj_mu.shape[0]
    s = 0.0
    for k in range(N):
        e = (traj_mu[k, :3] - gt[k, :3]).reshape(3, 1)
        Pk = traj_P[k, :3, :3]
        # numerically stable: solve instead of explicit inverse; add tiny jitter if needed
        try:
            u = np.linalg.solve(Pk, e)
        except np.linalg.LinAlgError:
            Pk = Pk + 1e-9 * np.eye(3)
            u = np.linalg.solve(Pk, e)
        s += float(e.T @ u)
    return s / (3*N)

def save_csv(path: str, rows: list, header: list):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)

def nees_conf_int_normal_approx(n_steps: int, d: int = 3, alpha: float = 0.05) -> tuple[float,float]:
    """
    For average NEES = (1/(dN)) * chi2_{dN}, mean=1, var=2/(dN).
    Use normal approx (good for moderate/large N): 1 ± z * sqrt(2/(dN)), z≈1.96 at 95%.
    """
    if n_steps <= 0:
        return (0.0, 0.0)
    z = 1.959963984540054  # Φ^{-1}(0.975)
    sigma = np.sqrt(2.0 / max(1, d*n_steps))
    lo = 1.0 - z * sigma
    hi = 1.0 + z * sigma
    return float(lo), float(hi)

def evaluate_and_save(target_mu_seq, target_P_seq, gt_target_seq, out_dir):
    rm = rmse_xyz(np.asarray(target_mu_seq)[:, :3], np.asarray(gt_target_seq)[:, :3])
    n = nees(np.asarray(target_mu_seq), np.asarray(target_P_seq), np.asarray(gt_target_seq))
    lo, hi = nees_conf_int_normal_approx(n_steps=len(target_mu_seq), d=3, alpha=0.05)
    ok = (n >= lo) and (n <= hi)
    print(f"[SWARM] Target RMSE (m): {rm}  NEES: {n:.3f}  95% band≈[{lo:.3f},{hi:.3f}]  {'OK' if ok else 'OUT-OF-BAND'}")

    # Save time series
    rows = []
    for k in range(len(target_mu_seq)):
        mu = target_mu_seq[k][:3]
        P = target_P_seq[k][:3, :3]
        rows.append([k, mu[0], mu[1], mu[2], P[0,0], P[1,1], P[2,2]])
    save_csv(os.path.join(out_dir, "target_estimate.csv"),
             rows, ["k", "x", "y", "z", "var_x", "var_y", "var_z"])

    save_csv(os.path.join(out_dir, "summary.csv"),
             [[rm["rmse_x"], rm["rmse_y"], rm["rmse_z"], rm["rmse_3d"], n]],
             ["rmse_x","rmse_y","rmse_z","rmse_3d","nees"])

    return rm, n
