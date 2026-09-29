"""Render the comparison video.

Left: map + live scan in the Leica frame, with every method's path.
Right: the FFT latitude slice (map vs scan), the phase-correlation peak,
and per-frame latency against the 10 Hz budget.

Usage: python render.py frames.npz results_dir gt_prism.tum out.mp4
"""
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

import odom
from evaluate import LABEL, MODES, REALTIME_MS, associate, load_est, umeyama

BG, FG, DIM = "#0e1116", "#e6e6e6", "#5b6370"
COL = {"fft": "#f0a93b", "fft_icp": "#39a0ff", "icp": "#a0a0a0", "icp3": "#c77ddb", "gt": "#ffffff"}
FPS = 20                 # 2x real time
PRE_ROLL_S = 1.5         # seconds shown before ground truth starts
MAP_VOXEL = 0.35
WIN_M = 8.0              # matches the FFT shift search window


def aligned_paths(res, t_gt, p_gt):
    """Each mode's prism path in the Leica frame, plus B's body->Leica alignment."""
    out, align_b = {}, None
    for m in MODES:
        t, p = load_est(f"{res}/traj_{m}.tum")
        ie, ig = associate(t, t_gt)
        R, tr = umeyama(p[ie], p_gt[ig])
        out[m] = p @ R.T + tr
        if m == "fft_icp":
            align_b = (R, tr)
    return out, align_b


def main():
    frames_path, res, gt_path, out_mp4 = sys.argv[1:5]
    stamps, frames = odom.load_frames(frames_path)
    g = np.loadtxt(gt_path)
    t_gt, p_gt = g[:, 0], g[:, 1:4]

    print("re-running method B with FFT internals recorded ...", flush=True)
    poses, _, video = odom.run(frames, odom.Mode.FFT_ICP, log_video=True)
    paths, (R_b, t_b) = aligned_paths(res, t_gt, p_gt)
    to_leica = lambda P: P @ R_b.T + t_b
    ms = {m: np.genfromtxt(f"{res}/timing_{m}.csv", delimiter=",", names=True) for m in MODES}
    ms = {m: (v["preproc"] + v["fft"] + v["icp"] + v["map"]) * 1e3 for m, v in ms.items()}
    err = {m: np.full(len(stamps), np.nan) for m in MODES}
    for m in MODES:
        ie, ig = associate(stamps, t_gt)
        err[m][ie] = np.linalg.norm(paths[m][ie] - p_gt[ig], axis=1)

    start = int(np.searchsorted(stamps, t_gt[0] - PRE_ROLL_S))
    world = odom.voxel(np.concatenate([to_leica(odom.apply(poses[i], odom.voxel(frames[i], 0.5)))
                                       for i in range(start, len(frames), 5)]), MAP_VOXEL)
    lo, hi = np.percentile(world[:, 2], [2, 98])
    world = world[(world[:, 2] > lo - 1) & (world[:, 2] < hi + 1)]

    plt.rcParams.update({"text.color": FG, "axes.labelcolor": FG, "xtick.color": DIM,
                         "ytick.color": DIM, "axes.edgecolor": DIM, "font.size": 11})
    fig = plt.figure(figsize=(19.2, 10.8), dpi=100, facecolor=BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.55, 1], height_ratios=[1, 1, 0.8],
                          left=0.04, right=0.98, top=0.9, bottom=0.06, wspace=0.12, hspace=0.32)
    ax_map = fig.add_subplot(gs[:, 0])
    ax_sl = fig.add_subplot(gs[0, 1])
    ax_pc = fig.add_subplot(gs[1, 1])
    ax_lat = fig.add_subplot(gs[2, 1])
    for a in (ax_map, ax_sl, ax_pc, ax_lat):
        a.set_facecolor(BG)

    fig.text(0.04, 0.955, "FFT-slice LiDAR odometry", fontsize=26, weight="bold")
    fig.text(0.04, 0.925, "NTU VIRAL eee_03  |  drone, 2x Ouster OS1-16 (horizontal + vertical)  |  "
             "LiDAR only, no IMU", fontsize=13, color=DIM)
    hud = fig.text(0.615, 0.95, "", fontsize=12.5, family="monospace", va="center")

    # Map panel: faint full map, bright live scan, trajectories
    ax_map.scatter(world[:, 0], world[:, 1], c=world[:, 2], s=2.2, cmap="viridis", alpha=0.8,
                   vmin=lo, vmax=hi, linewidths=0)
    live = ax_map.scatter([], [], s=1.6, c="#ff4fa0", alpha=0.75, linewidths=0)
    lines = {m: ax_map.plot([], [], color=COL[m], lw=2.2 if m == "fft_icp" else 1.4,
                            label=LABEL[m])[0] for m in MODES}
    gt_line, = ax_map.plot([], [], color=COL["gt"], lw=1.2, ls="--", label="Leica ground truth")
    drone, = ax_map.plot([], [], "o", ms=11, mfc="none", mec=COL["fft_icp"], mew=2.5)
    c0 = p_gt.mean(0)
    span = max(np.ptp(p_gt[:, 0]), np.ptp(p_gt[:, 1])) / 2 + 7
    ax_map.set(xlim=(c0[0] - span, c0[0] + span), ylim=(c0[1] - span, c0[1] + span), aspect="equal")
    ax_map.set_title("Map (height-coloured), live merged scan (pink), paths", loc="left", color=FG)
    ax_map.legend(loc="lower left", frameon=False, labelcolor=FG)

    # Slice panel: map latitude slice (grey) with scan on top (cyan)
    n = video[0]["map"].shape[0]
    ext = [-n * 0.125, n * 0.125, -n * 0.125, n * 0.125]
    im_map = ax_sl.imshow(np.zeros((n, n)), cmap="gray", extent=ext, origin="lower", vmin=0, vmax=1)
    im_scan = ax_sl.imshow(np.zeros((n, n, 4)), extent=ext, origin="lower")
    ax_sl.set_title("Latitude slice (XY): map grey, scan cyan", loc="left", color=FG)

    # Phase-correlation panel: surface inside the search window
    k = int(WIN_M / 0.25)
    im_pc = ax_pc.imshow(np.zeros((2 * k, 2 * k)), cmap="magma", extent=[-WIN_M, WIN_M, -WIN_M, WIN_M],
                         origin="lower")
    peak_dot, = ax_pc.plot([], [], "+", ms=18, mew=2, color="#39a0ff")
    ax_pc.set_title("Phase correlation over z-bands (peak = shift)", loc="left", color=FG)
    ax_pc.set_xlabel("m")

    # Latency strip
    hist = 120
    lat_lines = {m: ax_lat.plot([], [], color=COL[m], lw=1.6, label=LABEL[m])[0] for m in MODES}
    ax_lat.axhline(REALTIME_MS, color="#e05555", ls="--", lw=1)
    ax_lat.text(1, REALTIME_MS + 3, "10 Hz budget", color="#e05555", fontsize=10)
    ax_lat.set(xlim=(0, hist), ylim=(0, 220), ylabel="ms / frame")
    ax_lat.set_title("Per-frame latency (last 12 s)", loc="left", color=FG)
    ax_lat.legend(loc="upper right", frameon=False, labelcolor=FG, ncol=3, fontsize=9)

    writer = FFMpegWriter(fps=FPS, bitrate=9000, codec="libx264",
                          extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    idx = range(start, len(frames))
    print(f"rendering {len(idx)} frames ...", flush=True)
    with writer.saving(fig, out_mp4, dpi=100):
        for i in idx:
            scan_w = to_leica(odom.apply(poses[i], frames[i][::2]))
            live.set_offsets(scan_w[:, :2])
            for m in MODES:
                lines[m].set_data(paths[m][start:i + 1, 0], paths[m][start:i + 1, 1])
            done = t_gt <= stamps[i]
            gt_line.set_data(p_gt[done, 0], p_gt[done, 1])
            drone.set_data([paths["fft_icp"][i, 0]], [paths["fft_icp"][i, 1]])

            v = video[i - 1]
            mi = v["map"] / max(v["map"].max(), 1e-6)
            si = v["scan"] / max(v["scan"].max(), 1e-6)
            im_map.set_data(mi.T)
            rgba = np.zeros((n, n, 4))
            rgba[..., 1], rgba[..., 2], rgba[..., 3] = 0.9, 1.0, np.clip(si.T * 1.6, 0, 1)
            im_scan.set_data(rgba)

            corr = np.fft.fftshift(v["corr"])[n // 2 - k:n // 2 + k, n // 2 - k:n // 2 + k]
            # sqrt keeps the sidelobes visible next to the dominant peak
            corr = np.sqrt(np.clip(corr, 0, None))
            im_pc.set_data(corr.T)
            im_pc.set_clim(0, corr.max())
            peak_dot.set_data([v["shift"][0]], [v["shift"][1]])

            a = max(start, i - hist)
            for m in MODES:
                lat_lines[m].set_data(np.arange(i + 1 - a), ms[m][a:i + 1])

            e = {m: err[m][i] for m in MODES}
            hud.set_text(f"t {stamps[i] - t_gt[0]:6.1f} s   error now:  B {e['fft_icp']:5.2f} m   "
                         f"ICP {e['icp']:5.2f} m   FFT {e['fft']:5.2f} m\n"
                         f"FFT step: yaw {np.degrees(v['yaw']):+5.2f} deg  roll {np.degrees(v['roll']):+5.2f}  "
                         f"pitch {np.degrees(v['pitch']):+5.2f}  peak {v['peak']:.2f}")
            writer.grab_frame(facecolor=BG)
    print("wrote", out_mp4)


if __name__ == "__main__":
    main()
