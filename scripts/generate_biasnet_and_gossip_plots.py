#!/usr/bin/env python3
"""Generate thesis figures for BiasNet diagnostics and Gossip CI rounds."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Ellipse

sns.set_context("talk")
plt.rcParams["font.size"] = 10


def load_biasnet_dataframe(paths: list[Path], max_records: Optional[int] = None) -> pd.DataFrame:
    """Load BiasNet training samples from JSONL logs that concatenate objects."""
    decoder = json.JSONDecoder()
    records = []
    budget = max_records if max_records and max_records > 0 else None

    for path in paths:
        try:
            text = path.read_text()
        except Exception as exc:
            print(f"  ⚠ Failed to read {path}: {exc}")
            continue

        idx = 0
        length = len(text)
        while idx < length:
            while idx < length and text[idx].isspace():
                idx += 1
            if idx >= length:
                break
            try:
                sample, new_idx = decoder.raw_decode(text, idx)
            except json.JSONDecodeError as exc:
                print(f"  ⚠ JSON decode error in {path}: {exc}")
                break
            idx = new_idx

            feats = sample.get("features", [])
            if len(feats) < 12:
                continue
            meta = sample.get("meta", {})
            records.append(
                {
                    "bias": float(sample.get("bias", np.nan)),
                    "R_pair": float(feats[9]) if len(feats) > 9 else np.nan,
                    "m_eff": float(feats[10]) if len(feats) > 10 else np.nan,
                    "iqr": float(feats[11]) if len(feats) > 11 else np.nan,
                    "tracker": meta.get("tracker", "unknown"),
                    "exp": meta.get("exp", "unknown"),
                }
            )

            if budget is not None and len(records) >= budget:
                return pd.DataFrame(records)

    return pd.DataFrame(records)


def plot_biasnet_diagnostics(df: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    if df.empty:
        print("  ✗ No BiasNet samples available; skipping BiasNet plot.")
        return None

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["bias", "R_pair", "iqr", "tracker"])
    if df.empty:
        print("  ✗ BiasNet dataframe has only non-finite entries; skipping plot.")
        return None

    n_samples = len(df)
    n_experiments = int(df["exp"].nunique())
    trackers = sorted({str(t) for t in df["tracker"].unique()})

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))

    hb0 = axes[0].hexbin(df["R_pair"], df["bias"], gridsize=45, cmap="viridis", mincnt=5)
    fig.colorbar(hb0, ax=axes[0], label="Samples per bin")
    corr_rpair = df[["R_pair", "bias"]].corr().loc["R_pair", "bias"]
    axes[0].set_xlabel(r"$R^{\text{pair}}$ variance (m$^2$)", fontweight="bold")
    axes[0].set_ylabel("Observed bias (m)", fontweight="bold")
    axes[0].set_title("Higher measurement variance aligns with bias", fontweight="bold")
    if not np.isnan(corr_rpair):
        axes[0].text(0.04, 0.92, f"ρ = {corr_rpair:.2f}", transform=axes[0].transAxes,
                      fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    hb1 = axes[1].hexbin(df["iqr"], df["bias"], gridsize=45, cmap="magma", mincnt=5)
    fig.colorbar(hb1, ax=axes[1], label="Samples per bin")
    corr_iqr = df[["iqr", "bias"]].corr().loc["iqr", "bias"]
    axes[1].set_xlabel("Interquartile range (m)", fontweight="bold")
    axes[1].set_ylabel("Observed bias (m)", fontweight="bold")
    axes[1].set_title("Dispersion (IQR) rises with bias magnitude", fontweight="bold")
    if not np.isnan(corr_iqr):
        axes[1].text(0.04, 0.92, f"ρ = {corr_iqr:.2f}", transform=axes[1].transAxes,
                      fontsize=9, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))

    sns.boxplot(data=df, x="tracker", y="bias", ax=axes[2], palette="Set2")
    axes[2].axhline(0.0, color="#424242", linestyle="--", linewidth=1.0, alpha=0.7)
    axes[2].set_xlabel("Tracker ID", fontweight="bold")
    axes[2].set_ylabel("Observed bias (m)", fontweight="bold")
    axes[2].set_title("Hardware-specific offsets captured by BiasNet", fontweight="bold")
    axes[2].tick_params(axis="x", rotation=30)

    fig.suptitle(
        f"BiasNet training signals across {n_experiments} real experiments\n"
        f"Samples: {n_samples:,} — trackers: {', '.join(trackers)}",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "13_biasnet_feature_diagnostics.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {out_path}")
    return out_path


def confidence_ellipse(center: np.ndarray, covariance: np.ndarray, n_std: float = 1.0, **kwargs) -> Ellipse:
    vals, vecs = np.linalg.eigh(covariance)
    order = vals.argsort()[::-1]
    vals = np.maximum(vals[order], 0.0)
    vecs = vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    width, height = 2.0 * n_std * np.sqrt(vals)
    return Ellipse(xy=center, width=width, height=height, angle=angle, **kwargs)


def load_ci_snapshot(
    snaps_file: Path,
    weights_file: Path,
    min_mix: float = 1e-3,
    dominant_tracker: Optional[str] = None,
    dominant_threshold: float = 0.9,
    allow_extremes: bool = False,
    weight_target: Optional[float] = None,
    weight_tolerance: float = 0.15,
) -> Optional[dict]:
    if not snaps_file.exists() or not weights_file.exists():
        print("  ✗ Missing CI snapshot inputs; skipping Gossip CI plot.")
        return None

    with weights_file.open() as wf:
        reader = csv.DictReader(wf)
        weight_map = {float(row["timestamp"]): row for row in reader}

    best_snapshot: Optional[dict] = None
    best_error = np.inf
    best_weight_diff = np.inf

    with snaps_file.open() as sf:
        for line in sf:
            snap = json.loads(line)
            order = snap.get("order", [])
            if len(order) < 2:
                continue
            ts = float(snap["timestamp"])
            weight_row = weight_map.get(ts)
            if weight_row is None:
                continue
            weights = []
            valid = True
            for rid in order:
                try:
                    w_val = float(weight_row.get(f"w_{rid}", "nan"))
                except (TypeError, ValueError):
                    valid = False
                    break
                if not np.isfinite(w_val):
                    valid = False
                    break
                weights.append(w_val)
            if not valid:
                continue
            weights = np.asarray(weights, dtype=float)
            if np.any(weights < 0.0) or np.any(weights > 1.0):
                continue
            if not allow_extremes and (np.any(weights < min_mix) or np.any(weights > 1.0 - min_mix)):
                continue

            weight_diff = 0.0
            if dominant_tracker is not None:
                try:
                    dom_idx = order.index(dominant_tracker)
                except ValueError:
                    continue
                dom_weight = weights[dom_idx]
                if dom_weight < dominant_threshold:
                    continue
                if weight_target is not None:
                    weight_diff = abs(dom_weight - weight_target)
                    if weight_diff > weight_tolerance:
                        continue

            mus = np.asarray(snap["mus"], dtype=float)[:, :3]
            covs = np.asarray(snap["Ps"], dtype=float)[:, :3, :3]
            if np.allclose(mus, 0.0):
                continue
            gt = np.asarray(snap.get("gt_pos", [np.nan, np.nan, np.nan]), dtype=float)[:3]
            info_mats = np.linalg.inv(covs)
            info_sum = np.zeros_like(info_mats[0])
            info_mu = np.zeros(3)
            for w, info, mu in zip(weights, info_mats, mus):
                info_sum += w * info
                info_mu += w * info @ mu
            try:
                cov_ci = np.linalg.inv(info_sum)
            except np.linalg.LinAlgError:
                continue
            mu_ci = cov_ci @ info_mu
            err = float(np.linalg.norm(mu_ci - gt)) if np.all(np.isfinite(gt)) else np.inf

            candidate = {
                "timestamp": ts,
                "order": order,
                "weights": weights,
                "mus": mus,
                "covs": covs,
                "gt": gt,
                "mu_ci": mu_ci,
                "cov_ci": cov_ci,
                "error": err,
                "exp": snap.get("exp"),
            }
            if weight_target is not None and dominant_tracker is not None:
                if (weight_diff < best_weight_diff - 1e-6) or (
                    abs(weight_diff - best_weight_diff) <= 1e-6 and err < best_error
                ):
                    best_weight_diff = weight_diff
                    best_error = err
                    best_snapshot = candidate
            elif err < best_error:
                best_error = err
                best_snapshot = candidate

    if best_snapshot is None:
        print("  ✗ No suitable snapshot found for Gossip CI plot.")
    return best_snapshot


def simulate_gossip_ci(
    snapshot: dict,
    rounds: int = 3,
    p_link: float = 0.85,
    p_drop: float = 0.2,
    sharpen_eta: float = 0.15,
    seed: int = 21,
    mix_rate: float = 0.45,
) -> Optional[dict]:
    if snapshot is None:
        return None

    mus = np.asarray(snapshot.get("mus", []), dtype=float)
    covs = np.asarray(snapshot.get("covs", []), dtype=float)
    order = snapshot.get("order", [])
    if mus.size == 0 or covs.size == 0 or len(order) == 0:
        return None

    mus = mus[:, :3]
    covs = covs[:, :3, :3]
    n_nodes = len(order)
    dims = mus.shape[1]
    gt = np.asarray(snapshot.get("gt", [np.nan, np.nan, np.nan]), dtype=float)[:dims]

    info_mats = np.zeros_like(covs)
    info_vecs = np.zeros((n_nodes, dims))
    for idx in range(n_nodes):
        cov = covs[idx]
        jitter = 1e-6
        for _ in range(6):
            try:
                info = np.linalg.inv(cov)
                break
            except np.linalg.LinAlgError:
                cov = cov + np.eye(dims) * jitter
                jitter *= 10.0
        else:
            info = np.linalg.pinv(cov)
        mu = mus[idx]
        info_mats[idx] = info
        info_vecs[idx] = info @ mu

    y = np.hstack((info_mats.reshape(n_nodes, dims * dims), info_vecs))
    states = [{"label": "Round 0", "mu": mus.copy(), "cov": covs.copy()}]
    statuses = []

    rng = np.random.default_rng(seed)
    for round_idx in range(1, rounds + 1):
        active_edges = []
        W = np.eye(n_nodes)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if rng.random() < p_link:
                    if rng.random() > p_drop:
                        active_edges.append((i, j))
                        W[i, i] -= mix_rate
                        W[j, j] -= mix_rate
                        W[i, j] = mix_rate
                        W[j, i] = mix_rate
        y = W @ y
        mus_round = []
        covs_round = []
        for idx in range(n_nodes):
            info = y[idx, : dims * dims].reshape(dims, dims)
            info = 0.5 * (info + info.T)
            try:
                cov = np.linalg.inv(info)
            except np.linalg.LinAlgError:
                cov = np.linalg.pinv(info)
            info_vec = y[idx, dims * dims : dims * dims + dims]
            mu = cov @ info_vec
            mus_round.append(mu)
            covs_round.append(cov)
        states.append({"label": f"Round {round_idx}", "mu": np.array(mus_round), "cov": np.array(covs_round)})
        statuses.append({"round": round_idx, "active_edges": active_edges, "W": W if n_nodes == 2 else None})

    info_sum = np.zeros((dims, dims))
    info_vec_sum = np.zeros(dims)
    for state in states:
        pass
    for idx in range(n_nodes):
        cov = covs[idx]
        jitter = 1e-6
        for _ in range(6):
            try:
                info = np.linalg.inv(cov)
                break
            except np.linalg.LinAlgError:
                cov = cov + np.eye(dims) * jitter
                jitter *= 10.0
        else:
            info = np.linalg.pinv(cov)
        mu = mus[idx]
        info_sum += info
        info_vec_sum += info @ mu

    try:
        P_ci = np.linalg.inv(info_sum)
    except np.linalg.LinAlgError:
        info_sum = info_sum + np.eye(dims) * 1e-6
        P_ci = np.linalg.inv(info_sum)
    mu_ci = P_ci @ info_vec_sum

    J_sharp = (1.0 + sharpen_eta) * info_sum
    h_sharp = (1.0 + sharpen_eta) * info_vec_sum
    try:
        P_sharp = np.linalg.inv(J_sharp)
    except np.linalg.LinAlgError:
        J_sharp = J_sharp + np.eye(dims) * 1e-6
        P_sharp = np.linalg.inv(J_sharp)
    mu_sharp = P_sharp @ h_sharp

    rmse_ci = float(np.linalg.norm(mu_ci - gt)) if np.all(np.isfinite(gt)) else np.nan

    return {
        "states": states,
        "status": statuses,
        "ci": {"mu": mu_ci, "cov": P_ci, "rmse": rmse_ci},
        "sharpen": {"mu": mu_sharp, "cov": P_sharp, "eta": sharpen_eta},
        "config": {
            "rounds": rounds,
            "p_link": p_link,
            "p_drop": p_drop,
            "seed": seed,
            "mix_rate": mix_rate,
        },
        "gt": gt,
        "order": snapshot.get("order", []),
    }


def plot_gossip_ci_rounds(snapshot: dict, sim_result: dict, output_path: Path) -> Path:
    states = sim_result["states"]
    statuses = sim_result["status"]
    order = sim_result["order"]
    gt_xy = np.asarray(sim_result.get("gt", [np.nan, np.nan, np.nan]), dtype=float)[:2]

    colors = ["#d62728", "#1f77b4", "#9467bd", "#8c564b"]
    xy_points = [state["mu"][:, :2] for state in states]
    xy_points.append(sim_result["ci"]["mu"][:2].reshape(1, -1))
    xy_stack = np.vstack(xy_points)
    pad = 0.4
    xmin, ymin = np.min(xy_stack, axis=0) - pad
    xmax, ymax = np.max(xy_stack, axis=0) + pad

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    panel_titles = ["Round 0: Local beliefs"]
    for status in statuses:
        suffix = "link active" if status["active_edges"] else "link dropped"
        panel_titles.append(f"Round {status['round']}: {suffix}")
    panel_titles.append("Consensus → CI + sharpen")

    for ax_idx, ax in enumerate(axes):
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal", "box")
        ax.grid(alpha=0.3, linewidth=0.6)

        if ax_idx < len(states) - 1:
            state = states[ax_idx]
            ax.set_title(panel_titles[ax_idx], fontweight="bold")
            for node_idx, tracker in enumerate(order):
                color = colors[node_idx % len(colors)]
                mu_xy = state["mu"][node_idx][:2]
                cov_xy = state["cov"][node_idx][:2, :2]
                ell = confidence_ellipse(mu_xy, cov_xy, n_std=1.0, edgecolor=color, facecolor="none", linewidth=2.0)
                ax.add_patch(ell)
                ax.scatter(mu_xy[0], mu_xy[1], color=color, s=55)
                ax.text(mu_xy[0], mu_xy[1] + 0.08, tracker.upper(), color=color, ha="center", fontsize=8, fontweight="bold")

                if ax_idx > 0:
                    prev_state = states[ax_idx - 1]
                    prev_mu_xy = prev_state["mu"][node_idx][:2]
                    delta = mu_xy - prev_mu_xy
                    if np.linalg.norm(delta) > 1e-6:
                        ax.annotate("", xy=mu_xy, xytext=prev_mu_xy,
                                    arrowprops=dict(arrowstyle="->", color=color, linewidth=1.4, shrinkA=4, shrinkB=4))
                        ax.text(mu_xy[0] + 0.06, mu_xy[1] + 0.02, f"Δ={np.linalg.norm(delta)*100:.1f}cm",
                                color=color, fontsize=7, fontweight="bold")

            if ax_idx > 0:
                status = statuses[ax_idx - 1]
                if status["active_edges"]:
                    edge_labels = ", ".join(f"{order[i].upper()}↔{order[j].upper()}" for i, j in status["active_edges"])
                    note = f"Active edges: {edge_labels}"
                else:
                    note = "No messages received"
                W = status["W"]
                if W is not None:
                    note += f"\nW = [[{W[0,0]:.2f}, {W[0,1]:.2f}], [{W[1,0]:.2f}, {W[1,1]:.2f}]]"
                ax.text(0.02, 0.95, note, transform=ax.transAxes, ha="left", va="top",
                        fontsize=8, bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
        elif ax_idx == len(states) - 1:
            ax.set_title(panel_titles[ax_idx], fontweight="bold")
            final_state = states[-1]
            for node_idx, tracker in enumerate(order):
                color = colors[node_idx % len(colors)]
                mu_xy = final_state["mu"][node_idx][:2]
                cov_xy = final_state["cov"][node_idx][:2, :2]
                ell = confidence_ellipse(mu_xy, cov_xy, n_std=1.0, edgecolor=color, facecolor="none", linewidth=1.5, linestyle=":")
                ax.add_patch(ell)
                ax.scatter(mu_xy[0], mu_xy[1], color=color, s=45, alpha=0.7)

            mu_ci = sim_result["ci"]["mu"][:2]
            cov_ci = sim_result["ci"]["cov"][:2, :2]
            ell_ci = confidence_ellipse(mu_ci, cov_ci, n_std=1.0, edgecolor="#2ca02c", facecolor="none", linewidth=2.2)
            ax.add_patch(ell_ci)
            ax.scatter(mu_ci[0], mu_ci[1], color="#2ca02c", s=65)

            mu_sharp = sim_result["sharpen"]["mu"][:2]
            cov_sharp = sim_result["sharpen"]["cov"][:2, :2]
            ell_sharp = confidence_ellipse(mu_sharp, cov_sharp, n_std=1.0, edgecolor="#ff7f0e", facecolor="none", linewidth=2.2, linestyle="--")
            ax.add_patch(ell_sharp)
            ax.scatter(mu_sharp[0], mu_sharp[1], color="#ff7f0e", s=65)

            eta = sim_result["sharpen"]["eta"]
            rmse = sim_result["ci"]["rmse"]
            ax.text(0.02, 0.95, f"CI RMSE = {rmse:.2f} m\nSharpen η = {eta:.2f}",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.75, edgecolor="none"))
        else:
            ax.axis("off")

        if np.all(np.isfinite(gt_xy)):
            ax.scatter(gt_xy[0], gt_xy[1], marker="x", color="black", s=70)
            if ax_idx == 0:
                ax.text(gt_xy[0], gt_xy[1] - 0.12, "Ground truth", color="black", ha="center", fontsize=8)

    sim_cfg = sim_result["config"]
    ts = snapshot.get("timestamp", 0.0)
    exp_label = str(snapshot.get("exp", "Unknown Experiment")).replace("_ifo003", "")
    fig.suptitle(
        f"Gossip CI Rounds (Real Run)\n{exp_label} @ t={ts:.2f}s • rounds={sim_cfg['rounds']},"
        f" p_link={sim_cfg['p_link']:.2f}, p_drop={sim_cfg['p_drop']:.2f}, mix={sim_cfg['mix_rate']:.2f}",
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {output_path}")
    return output_path


def main() -> None:
    output_dir = Path("thesis_plots")
    output_dir.mkdir(exist_ok=True)

    print("=== BiasNet Feature Diagnostics ===")
    bias_paths = [p for p in Path("runs").rglob("bias_samples.jsonl") if p.is_file() and p.stat().st_size > 0]
    bias_df = load_biasnet_dataframe(bias_paths, max_records=120_000)
    plot_biasnet_diagnostics(bias_df, output_dir)

    print("\n=== Gossip CI Rounds ===")
    ci_run_dir = Path("runs/20251005_ci_snapshot_random2_0/default_3_random2_0_ifo003")
    snaps_path = ci_run_dir / "fusion_snaps.jsonl"
    weights_path = ci_run_dir / "fusion_weights.csv"
    ci_snapshot = load_ci_snapshot(snaps_path, weights_path)
    if ci_snapshot:
        gossip_sim = simulate_gossip_ci(ci_snapshot)
        if gossip_sim:
            plot_gossip_ci_rounds(ci_snapshot, gossip_sim, output_dir / "11d_gossip_ci_rounds.png")
        else:
            print("  ✗ Gossip CI simulation failed.")


if __name__ == "__main__":
    main()
