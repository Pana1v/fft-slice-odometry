"""Split-screen GIF: vanilla ICP (3 it) vs FSICP (FFT seed + the same 3 ICP iterations).

Each side builds its own map from its own poses, so pose error shows up as
ghost walls. Both runs start from the same pose, so one transform (fitted on
FFT + ICP) places both in the Leica frame and drift grows from the start.

Usage: python gif_race.py frames.npz results_dir gt_prism.tum out.gif
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

import odom
from evaluate import T_B_PRISM, associate, umeyama

BG, FG, DIM = "#0d1117", "#e8e8e8", "#6b7380"
GOOD, WARN, BAD = "#3ddc84", "#f5b942", "#ff5a5a"
SIDES = [("icp3", "Vanilla ICP", "#c77ddb"),
         ("fft_icp", "FSICP", "#39a0ff")]
RES = 0.08            # m per map pixel
WALL_MIN, WALL_MAX = 0.6, 9.0   # m above the ground, keeps walls, drops the floor
FPS = 8
PRE_ROLL_S = 1.5
WIDTH_PX = 1100


def load_poses(path):
    d = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(d), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(d[:, 4:8]).as_matrix()
    T[:, :3, 3] = d[:, 1:4]
    return d[:, 0], T


def err_color(e):
    if not np.isfinite(e) or e < 0.25:
        return GOOD
    return WARN if e < 1.0 else BAD


def main():
    frames_path, res, gt_path, out_gif = sys.argv[1:5]
    stamps, frames = odom.load_frames(frames_path)
    g = np.loadtxt(gt_path)
    t_gt, p_gt = g[:, 0], g[:, 1:4]

    poses = {m: load_poses(f"{res}/traj_{m}.tum")[1] for m, *_ in SIDES}
    prism = {m: poses[m][:, :3, 3] + poses[m][:, :3, :3] @ T_B_PRISM for m in poses}

    # One shared odometry -> Leica transform, fitted on the FSICP run
    ie, ig = associate(stamps, t_gt)
    R, t = umeyama(prism["fft_icp"][ie], p_gt[ig])
    to_w = lambda P: P @ R.T + t
    path = {m: to_w(prism[m]) for m in poses}
    err = {m: np.full(len(stamps), np.nan) for m in poses}
    for m in poses:
        err[m][ie] = np.linalg.norm(path[m][ie] - p_gt[ig], axis=1)
    ms = {m: np.genfromtxt(f"{res}/timing_{m}.csv", delimiter=",", names=True) for m in poses}
    ms = {m: (v["preproc"] + v["fft"] + v["icp"] + v["map"]) * 1e3 for m, v in ms.items()}

    ground = p_gt[0, 2] - 0.45        # prism sits ~0.45 m above the ground when parked
    c0 = p_gt.mean(axis=0)
    half = max(np.ptp(p_gt[:, 0]), np.ptp(p_gt[:, 1])) / 2 + 9.0
    n = int(2 * half / RES)
    acc = {m: np.zeros((n, n), np.float32) for m in poses}

    def add(m, i):
        pw = to_w(odom.apply(poses[m][i], frames[i]))
        h = pw[:, 2] - ground
        pw = pw[(h > WALL_MIN) & (h < WALL_MAX)]
        ij = np.floor((pw[:, :2] - (c0[:2] - half)) / RES).astype(int)
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        np.add.at(acc[m], (ij[ok, 1], ij[ok, 0]), 1.0)
        return pw

    start = int(np.searchsorted(stamps, t_gt[0] - PRE_ROLL_S))
    for m in poses:
        for i in range(start):
            add(m, i)

    plt.rcParams.update({"text.color": FG, "font.size": 12, "axes.edgecolor": "#2a313b"})
    fig = plt.figure(figsize=(12.8, 7.6), dpi=100, facecolor=BG)
    fig.text(0.03, 0.945, "Vanilla ICP vs FSICP", fontsize=22, weight="bold")
    fig.text(0.97, 0.945, "NTU VIRAL eee_03", fontsize=12, color=DIM, ha="right")

    ext = [c0[0] - half, c0[0] + half, c0[1] - half, c0[1] + half]
    art = {}
    for k, (m, name, col) in enumerate(SIDES):
        ax = fig.add_axes([0.03 + k * 0.485, 0.24, 0.455, 0.66])
        ax.set_facecolor(BG)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((n, n)), extent=ext, origin="lower", cmap="bone", vmin=0, vmax=1)
        gt_l, = ax.plot([], [], color="white", lw=1.3, ls="--", alpha=0.8)
        est_l, = ax.plot([], [], color=col, lw=2.6)
        scan = ax.scatter([], [], s=1.0, c="#7ef9ff", alpha=0.55, linewidths=0)
        dot, = ax.plot([], [], "o", ms=9, color=col, mec="white", mew=1.5)
        head, = ax.plot([], [], color="white", lw=2)
        ax.set_xlim(ext[:2]); ax.set_ylim(ext[2:])
        ax.text(0.02, 0.97, name, transform=ax.transAxes, fontsize=18, weight="bold", color=col, va="top")
        e_txt = ax.text(0.98, 0.03, "", transform=ax.transAxes, fontsize=26, weight="bold", ha="right",
                        va="bottom")
        ax.plot([ext[0] + 1, ext[0] + 6], [ext[2] + 1.2] * 2, color=FG, lw=3)
        ax.text(ext[0] + 3.5, ext[2] + 1.6, "5 m", ha="center", fontsize=10)
        art[m] = (im, gt_l, est_l, scan, dot, head, e_txt)

    # Bottom: error over time for both, plus compute per frame
    ax_e = fig.add_axes([0.06, 0.06, 0.56, 0.13])
    ax_e.set_facecolor(BG)
    t_rel = stamps - t_gt[0]
    e_lines = {m: ax_e.plot([], [], color=col, lw=2.2)[0] for m, _, col in SIDES}
    ax_e.set_xlim(t_rel[start], t_rel[-1])
    ax_e.set_ylim(0, np.nanmax([np.nanmax(err[m]) for m in poses]) * 1.1)
    ax_e.set_ylabel("error [m]", color=DIM, fontsize=10)
    ax_e.set_xlabel("time [s]", color=DIM, fontsize=10)
    ax_e.tick_params(colors=DIM, labelsize=9)
    ax_e.grid(alpha=0.15)

    ax_c = fig.add_axes([0.70, 0.06, 0.27, 0.13])
    ax_c.set_facecolor(BG)
    bars = ax_c.barh([0, 1], [0, 0], color=[s[2] for s in SIDES], height=0.55)
    ax_c.set_yticks([0, 1], [s[1] for s in SIDES], color=FG, fontsize=10)
    ax_c.set_xlim(0, 60)
    ax_c.invert_yaxis()
    ax_c.tick_params(colors=DIM, labelsize=9)
    ax_c.set_title("ms per scan", color=DIM, fontsize=10, loc="left")
    c_txt = [ax_c.text(0, y, "", va="center", fontsize=10, color=FG) for y in (0, 1)]

    with tempfile.TemporaryDirectory() as tmp:
        k = 0
        for i in range(start, len(frames)):
            for j, (m, *_ ) in enumerate(SIDES):
                pw = add(m, i)
                im, gt_l, est_l, scan, dot, head, e_txt = art[m]
                im.set_data(np.clip(np.log1p(acc[m]) / 3.5, 0, 1))
                done = t_gt <= stamps[i]
                gt_l.set_data(p_gt[done, 0], p_gt[done, 1])
                est_l.set_data(path[m][start:i + 1, 0], path[m][start:i + 1, 1])
                scan.set_offsets(pw[::3, :2])
                p = path[m][i]
                dot.set_data([p[0]], [p[1]])
                fwd = R @ poses[m][i][:3, 0]
                head.set_data([p[0], p[0] + 1.6 * fwd[0]], [p[1], p[1] + 1.6 * fwd[1]])
                e = err[m][i]
                e_txt.set_text("--" if not np.isfinite(e) else f"{e:.2f} m")
                e_txt.set_color(err_color(e))
                e_lines[m].set_data(t_rel[start:i + 1], err[m][start:i + 1])

                avg = np.mean(ms[m][start:i + 1])
                bars[j].set_width(avg)
                c_txt[j].set_x(avg + 1)
                c_txt[j].set_text(f"{avg:.0f}")
            fig.savefig(f"{tmp}/f{k:04d}.png", facecolor=BG)
            k += 1

        # Hold the last frame so the final state can be read
        last = f"{tmp}/f{k - 1:04d}.png"
        for h in range(FPS * 3):
            Path(f"{tmp}/f{k + h:04d}.png").symlink_to(last)

        vf = (f"fps={FPS},scale={WIDTH_PX}:-1:flags=lanczos,split[a][b];"
              "[a]palettegen=max_colors=160:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                        "-vf", vf, "-loop", "0", out_gif], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(FPS), "-i", f"{tmp}/f%04d.png",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
                        str(Path(out_gif).with_suffix(".mp4"))], check=True)
    print("wrote", out_gif, k, "frames")


if __name__ == "__main__":
    main()
