"""Translational ATE/RPE against the Leica prism track, plus speed plots.

Usage: python evaluate.py results_dir gt_prism.tum [--published published.json]

Ground truth is position only, so only translation errors exist here.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

# Body -> prism, from the NTU VIRAL evaluation tutorial
T_B_PRISM = np.array([-0.293656, -0.012288, -0.273095])
ASSOC_TOL = 0.03       # s; GT is 20 Hz, scans 10 Hz
RPE_DIST = 10.0        # m of travelled path per RPE segment
REALTIME_MS = 100.0    # 10 Hz sensor
MODES = ["fft", "fft_icp", "icp", "icp3"]
LABEL = {"fft": "Freq Domain Only", "fft_icp": "FSICP (FFT seed + 3 ICP it)", "icp": "Vanilla ICP (15 it)",
         "icp3": "Vanilla ICP (3 it)"}
COLOR = {"fft": "#e8a33d", "fft_icp": "#2f7fc1", "icp": "#8a8a8a", "icp3": "#b05fc4", "gt": "#222222"}


def load_est(path):
    d = np.loadtxt(path)
    R = Rotation.from_quat(d[:, 4:8]).as_matrix()
    prism = d[:, 1:4] + R @ T_B_PRISM
    return d[:, 0], prism


def associate(t_est, t_gt):
    j = np.clip(np.searchsorted(t_gt, t_est), 1, len(t_gt) - 1)
    j = np.where(np.abs(t_gt[j - 1] - t_est) < np.abs(t_gt[j] - t_est), j - 1, j)
    ok = np.abs(t_gt[j] - t_est) < ASSOC_TOL
    return np.nonzero(ok)[0], j[ok]


def umeyama(src, dst):
    """Rigid (no scale) alignment minimizing ||R src + t - dst||."""
    ms, md = src.mean(0), dst.mean(0)
    U, _, Vt = np.linalg.svd((dst - md).T @ (src - ms))
    S = np.eye(3)
    S[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ S @ Vt
    return R, md - R @ ms


def rpe(est, gt):
    """Drift over segments of RPE_DIST travelled metres, in the aligned frame."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(gt, axis=0), axis=1))]
    j = np.searchsorted(s, s + RPE_DIST)
    i = np.nonzero(j < len(s))[0]
    e = (est[j[i]] - est[i]) - (gt[j[i]] - gt[i])
    return np.linalg.norm(e, axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("gt")
    ap.add_argument("--published")
    args = ap.parse_args()
    res = Path(args.results)
    plots = res / "plots"
    plots.mkdir(exist_ok=True)

    g = np.loadtxt(args.gt)
    t_gt, p_gt = g[:, 0], g[:, 1:4]

    runs, metrics = {}, {}
    for m in MODES:
        t, p = load_est(res / f"traj_{m}.tum")
        ie, ig = associate(t, t_gt)
        R, tr = umeyama(p[ie], p_gt[ig])
        pa = p @ R.T + tr
        err = np.linalg.norm(pa[ie] - p_gt[ig], axis=1)
        r = rpe(pa[ie], p_gt[ig])

        # Latency over the GT-matched frames only, the same window the error covers
        tim = np.genfromtxt(res / f"timing_{m}.csv", delimiter=",", names=True)[ie]
        ms = (tim["preproc"] + tim["fft"] + tim["icp"] + tim["map"]) * 1e3
        runs[m] = dict(t=t, p=pa, ie=ie, ig=ig, err=err, ms=ms, tim=tim)
        metrics[m] = {
            "ate_rmse_m": float(np.sqrt(np.mean(err ** 2))),
            "ate_max_m": float(err.max()),
            "rpe_mean_m_per_10m": float(r.mean()) if len(r) else None,
            "latency_ms_p50": float(np.percentile(ms, 50)),
            "latency_ms_p95": float(np.percentile(ms, 95)),
            "latency_ms_p99": float(np.percentile(ms, 99)),
            "mean_ms": {k: float(tim[k].mean() * 1e3) for k in ("preproc", "fft", "icp", "map")},
            "frames": int(len(t)),
            "gt_matched": int(len(ie)),
            "gt_duration_s": float(t_gt[ig[-1]] - t_gt[ig[0]]),
        }

    published = json.load(open(args.published)) if args.published else {}
    json.dump({"ours": metrics, "published": published}, open(res / "metrics.json", "w"), indent=2)

    t0 = t_gt[0]

    # 1. Trajectories, top and side
    fig, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.3, 1]})
    ax[0].plot(p_gt[:, 0], p_gt[:, 1], color=COLOR["gt"], lw=2.5, label="Leica ground truth")
    for m in MODES:
        ie, p = runs[m]["ie"], runs[m]["p"]
        ax[0].plot(p[ie, 0], p[ie, 1], color=COLOR[m], lw=1.4, label=LABEL[m])
        ax[1].plot(t_gt[runs[m]["ig"]] - t0, p[ie, 2], color=COLOR[m], lw=1.4)
    ax[1].plot(t_gt - t0, p_gt[:, 2], color=COLOR["gt"], lw=2.5)
    ax[0].set(title="Top view (prism position, aligned)", xlabel="x [m]", ylabel="y [m]", aspect="equal")
    ax[1].set(title="Height", xlabel="time since GT start [s]", ylabel="z [m]")
    ax[0].legend(frameon=False)
    for a in ax:
        a.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "trajectory.png", dpi=110)

    # 2. ATE over time
    fig, ax = plt.subplots(figsize=(12, 4))
    for m in MODES:
        ax.plot(t_gt[runs[m]["ig"]] - t0, runs[m]["err"], color=COLOR[m], lw=1.4,
                label=f"{LABEL[m]}  RMSE {metrics[m]['ate_rmse_m']:.3f} m")
    ax.set(title="Absolute position error vs Leica (translation only, no orientation GT)",
           xlabel="time since GT start [s]", ylabel="error [m]")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "ate_time.png", dpi=110)

    # 3. Latency distribution
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    bins = np.linspace(0, max(np.percentile(runs[m]["ms"], 99.5) for m in MODES), 60)
    for m in MODES:
        ms = runs[m]["ms"]
        ax[0].hist(ms, bins=bins, color=COLOR[m], alpha=0.55, label=LABEL[m])
        xs = np.sort(ms)
        ax[1].plot(xs, np.arange(1, len(xs) + 1) / len(xs), color=COLOR[m], lw=2,
                   label=f"{LABEL[m]}  p50 {metrics[m]['latency_ms_p50']:.0f} / p95 {metrics[m]['latency_ms_p95']:.0f} ms")
    for a in ax:
        a.axvline(REALTIME_MS, color="#c0392b", ls="--", lw=1)
        a.grid(alpha=0.25)
    ax[0].set(title="Per-frame latency", xlabel="ms", ylabel="frames")
    ax[1].set(title="Latency CDF (dashed: 10 Hz real-time budget)", xlabel="ms", ylabel="fraction of frames")
    ax[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots / "latency.png", dpi=110)

    # 4. Stage breakdown
    fig, ax = plt.subplots(figsize=(8, 4.5))
    stages = [("preproc", "#b9c6d2"), ("fft", "#e8a33d"), ("icp", "#2f7fc1"), ("map", "#7a9a5a")]
    left = np.zeros(len(MODES))
    for k, c in stages:
        v = np.array([metrics[m]["mean_ms"][k] for m in MODES])
        ax.barh([LABEL[m] for m in MODES], v, left=left, color=c, label=k)
        left += v
    ax.set(title="Mean time per frame by stage (map = keyframe updates, amortized)", xlabel="ms")
    ax.legend(frameon=False, ncol=4, loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plots / "stages.png", dpi=110)

    # 5. Accuracy vs speed, with published rows
    fig, ax = plt.subplots(figsize=(9, 4.5))
    names = [LABEL[m] for m in MODES] + list(published)
    vals = [metrics[m]["ate_rmse_m"] for m in MODES] + [v["ate_m"] for v in published.values()]
    cols = [COLOR[m] for m in MODES] + ["#c9c9c9"] * len(published)
    ax.barh(names, vals, color=cols)
    for y, v in enumerate(vals):
        ax.text(v, y, f" {v:.3f}", va="center")
    dur = metrics["fft_icp"]["gt_duration_s"]
    title = f"ATE RMSE [m]  |  LiDAR only, {dur:.0f} s of GT"
    if published:
        title += "  |  grey: published, full 181 s, uses IMU"
    ax.set(title=title, xlabel="m")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(plots / "ate_bars.png", dpi=110)

    for m in MODES:
        x = metrics[m]
        print(f"{m:8s} ATE rmse {x['ate_rmse_m']:.3f} max {x['ate_max_m']:.3f} | RPE/10m {x['rpe_mean_m_per_10m']} | "
              f"p50 {x['latency_ms_p50']:.1f} p95 {x['latency_ms_p95']:.1f} ms | gt {x['gt_matched']} over {x['gt_duration_s']:.0f} s")


if __name__ == "__main__":
    main()
