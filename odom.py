"""LiDAR odometry with three front-ends sharing one local map:

  fft      : FFT slice registration only           (ablation A)
  fft_icp  : FFT coarse + few ICP iterations       (method B)
  icp      : constant-velocity init + full ICP     (same-machine control)
  icp3     : constant-velocity init + same 3 ICP   (isolates what the FFT seed buys)

Usage: python odom.py frames.npz out_dir [--modes fft fft_icp icp]
"""
import argparse
import time
from enum import Enum
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from fftreg import SliceGrid, rot2


class Mode(str, Enum):
    FFT = "fft"
    FFT_ICP = "fft_icp"
    ICP = "icp"
    ICP3 = "icp3"


SRC_VOXEL = 0.4        # registration points
MAP_VOXEL = 0.25
MAP_RADIUS = 40.0
KEY_DIST = 1.0         # m moved before the map is updated
KEY_ANGLE = np.deg2rad(10.0)
KEY_FRAMES = 10
MAX_CORR = 1.0         # m, ICP correspondence gate
ICP_ITERS = {Mode.FFT_ICP: 3, Mode.ICP: 15, Mode.ICP3: 3}
ICP_EPS = 1e-4
NORMAL_K = 15

# Latitude slices: z-band edges relative to the keyframe (XY images)
Z_EDGES = np.array([-30.0, -0.8, 1.5, 30.0])
# Longitudinal slices: band edges across the other horizontal axis (XZ and YZ images)
SIDE_EDGES = np.array([-30.0, -8.0, -2.5, 2.5, 8.0, 30.0])
SIDE_PIX = 256
SIDE_RES = 0.25


def voxel_idx(pts, size):
    """Index of the first point in each voxel, so earlier points win."""
    keys = np.floor(pts / size).astype(np.int64)
    keys -= keys.min(axis=0)
    span = keys.max(axis=0) + 1
    flat = (keys[:, 0] * span[1] + keys[:, 1]) * span[2] + keys[:, 2]
    return np.unique(flat, return_index=True)[1]


def voxel(pts, size):
    return pts[voxel_idx(pts, size)]


def pca_normals(q, tree):
    _, nn = tree.query(q, k=NORMAL_K, workers=-1)
    nb = tree.data[nn]
    d = nb - nb.mean(axis=1, keepdims=True)
    return np.linalg.eigh(np.einsum("mki,mkj->mij", d, d))[1][:, :, 0]


def apply(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]


def exp_se3(x):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(x[:3]).as_matrix()
    T[:3, 3] = x[3:]
    return T


def about(R3, c, t):
    """4x4 for p -> R (p - c) + c + t."""
    T = np.eye(4)
    T[:3, :3] = R3
    T[:3, 3] = c + t - R3 @ c
    return T


def plane_rot(axes, a):
    """3x3 rotating the (u, v) coordinate pair by angle a."""
    R = np.eye(3)
    u, v = axes
    R[np.ix_([u, v], [u, v])] = rot2(a)
    return R


class LocalMap:
    def __init__(self):
        self.pts = np.zeros((0, 3))
        self.normals = np.zeros((0, 3))
        self.tree = None

    def update(self, scan_w, center, mode):
        pts = np.concatenate([self.pts, scan_w])
        nrm = np.concatenate([self.normals, np.full((len(scan_w), 3), np.nan)])
        keep = np.linalg.norm(pts - center, axis=1) < MAP_RADIUS
        pts, nrm = pts[keep], nrm[keep]

        # Old points come first, so they keep their voxel and their normal
        idx = voxel_idx(pts, MAP_VOXEL)
        self.pts, self.normals = pts[idx], nrm[idx]
        if mode is Mode.FFT:
            return

        self.tree = cKDTree(self.pts)
        new = np.isnan(self.normals[:, 0])
        self.normals[new] = pca_normals(self.pts[new], self.tree)


class FFTFront:
    """Coarse world-frame correction from slice images of scan vs map.

    Map spectra are cached once per keyframe in a grid centred on the
    keyframe, so each frame only rasterizes the scan.
    """

    def __init__(self, record=False):
        self.g = SliceGrid()
        # Separate grid so side views can be tuned apart; 128 px at 0.4 m doubled roll error
        self.gs = SliceGrid(n=SIDE_PIX, res=SIDE_RES)
        self.record = record
        self.last = {}

    def _spec(self, rel, img_axes, band_axis, edges, g=None):
        g = g or self.g
        band = np.searchsorted(edges, rel[:, band_axis]) - 1
        imgs = g.render(rel[:, img_axes], band, len(edges) - 1)
        return g.spectra(imgs), band

    def prepare(self, lmap, ck):
        self.ck = ck.copy()
        rel = lmap.pts - ck
        rel = rel[np.all(np.abs(rel) < self.g.half, axis=1)]
        self.F_xy, _ = self._spec(rel, [0, 1], 2, Z_EDGES)
        self.A_xy = self.g.lp_spectrum(self.F_xy)
        self.F_xz, _ = self._spec(rel, [0, 2], 1, SIDE_EDGES, self.gs)
        self.F_yz, _ = self._spec(rel, [1, 2], 0, SIDE_EDGES, self.gs)
        if self.record:
            self.map_img = self.g.render(rel[:, :2])[0]

    def _tilt(self, F_ref, rel, band_axis, img_axes, c):
        """Fit per-band vertical shift against band lever arm: dz_k = dz0 + slope * lever_k."""
        F_mov, band = self._spec(rel, img_axes, band_axis, SIDE_EDGES, self.gs)
        n_b = len(SIDE_EDGES) - 1
        ok = (band >= 0) & (band < n_b)
        cnt = np.bincount(band[ok], minlength=n_b)
        lever = np.bincount(band[ok], weights=rel[ok, band_axis], minlength=n_b) / np.maximum(cnt, 1)
        lever -= c[band_axis]

        shifts, peaks = self.gs.phase_corr_each(F_ref, F_mov)
        use = cnt >= 30
        if not use.any():
            return 0.0, 0.0
        lever, dz, w = lever[use], shifts[use, 1], peaks[use]
        if use.sum() < 2 or np.ptp(lever) < 2.0:
            return float(np.average(dz, weights=w)), 0.0
        A = np.c_[np.ones_like(lever), lever] * w[:, None]
        dz0, slope = np.linalg.lstsq(A, dz * w, rcond=None)[0]
        return dz0, slope

    def correct(self, T_pred, src, lmap):
        g = self.g
        c = T_pred[:3, 3] - self.ck
        rel = apply(T_pred, src) - self.ck

        # Latitude slices (XY over z-bands): yaw by Fourier-Mellin, then x, y
        a = g.rot_fm(None, self._spec(rel, [0, 1], 2, Z_EDGES)[0], A=self.A_xy)
        Rz = plane_rot([0, 1], -a)
        rel = (rel - c) @ Rz.T + c
        s_xy, peak, corr = g.phase_corr(self.F_xy, self._spec(rel, [0, 1], 2, Z_EDGES)[0])
        rel[:, :2] -= s_xy
        c2 = c.copy()
        c2[:2] -= s_xy

        # Longitudinal slices: a roll lifts points by roll*y, a pitch by -pitch*x
        dz_a, roll = self._tilt(self.F_xz, rel, 1, [0, 2], c2)
        dz_b, slope = self._tilt(self.F_yz, rel, 0, [1, 2], c2)
        pitch = -slope
        dz = 0.5 * (dz_a + dz_b)

        # World-frame correction, composed in the order applied
        T_yaw = about(Rz, c + self.ck, np.zeros(3))
        T_xy = about(np.eye(3), np.zeros(3), np.r_[-s_xy, 0.0])
        R_tilt = plane_rot([1, 2], -roll) @ plane_rot([2, 0], -pitch)
        T_tilt = about(R_tilt, c2 + self.ck, np.array([0.0, 0.0, -dz]))

        if self.record:
            self.last = {"map": self.map_img, "scan": g.render(rel[:, :2])[0], "corr": corr,
                         "peak": peak, "yaw": a, "shift": s_xy, "roll": roll, "pitch": pitch}
        return T_tilt @ T_xy @ T_yaw @ T_pred


def icp(T, src, lmap, iters):
    """Point-to-plane Gauss-Newton with a Geman-McClure style weight."""
    for k in range(iters):
        p = apply(T, src)
        d, idx = lmap.tree.query(p, distance_upper_bound=MAX_CORR, workers=-1)
        ok = np.isfinite(d)
        if ok.sum() < 50:
            return T, k + 1
        p, q, n = p[ok], lmap.pts[idx[ok]], lmap.normals[idx[ok]]

        r = np.einsum("ij,ij->i", p - q, n)
        J = np.hstack([np.cross(p, n), n])
        w = 1.0 / (1.0 + (r / 0.2) ** 2) ** 2
        H = J.T @ (J * w[:, None])
        b = J.T @ (w * r)
        x = -np.linalg.solve(H + 1e-9 * np.eye(6), b)
        T = exp_se3(x) @ T
        if np.linalg.norm(x) < ICP_EPS:
            return T, k + 1
    return T, iters


def run(frames, mode, log_video=False):
    lmap = LocalMap()
    fft = FFTFront(record=log_video)
    poses, timing, video = [], [], []
    T_prev = T_prev2 = np.eye(4)
    T_key = None

    for i, scan in enumerate(frames):
        t0 = time.perf_counter()
        src = voxel(scan, SRC_VOXEL)
        t1 = time.perf_counter()

        if i == 0:
            T = np.eye(4)
            t2 = t3 = t1
            n_it = 0
        else:
            T = T_prev @ np.linalg.inv(T_prev2) @ T_prev
            if mode in (Mode.FFT, Mode.FFT_ICP):
                T = fft.correct(T, src, lmap)
            t2 = time.perf_counter()
            n_it = 0
            if mode is not Mode.FFT:
                T, n_it = icp(T, src, lmap, ICP_ITERS[mode])
            t3 = time.perf_counter()

        is_key = T_key is None or i % KEY_FRAMES == 0
        if T_key is not None and not is_key:
            dT = np.linalg.inv(T_key) @ T
            ang = np.linalg.norm(Rotation.from_matrix(dT[:3, :3]).as_rotvec())
            is_key = np.linalg.norm(dT[:3, 3]) > KEY_DIST or ang > KEY_ANGLE
        if is_key:
            lmap.update(apply(T, voxel(scan, MAP_VOXEL)), T[:3, 3], mode)
            if mode in (Mode.FFT, Mode.FFT_ICP):
                fft.prepare(lmap, T[:3, 3])
            T_key = T
        t4 = time.perf_counter()

        poses.append(T)
        timing.append((t1 - t0, t2 - t1, t3 - t2, t4 - t3, n_it, is_key))
        if log_video and mode in (Mode.FFT, Mode.FFT_ICP) and i > 0:
            video.append(fft.last)
        T_prev2, T_prev = T_prev, T

    return np.array(poses), np.array(timing), video


def save_tum(path, stamps, poses):
    q = Rotation.from_matrix(poses[:, :3, :3]).as_quat()
    rows = np.hstack([stamps[:, None], poses[:, :3, 3], q])
    np.savetxt(path, rows, fmt="%.9f")


def load_frames(path):
    z = np.load(path)
    off = z["offsets"]
    pts = z["points"].astype(np.float64)
    return z["stamps"], [pts[off[i]:off[i + 1]] for i in range(len(off) - 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frames")
    ap.add_argument("out")
    ap.add_argument("--modes", nargs="+", default=[m.value for m in Mode])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    stamps, frames = load_frames(args.frames)
    if args.limit:
        stamps, frames = stamps[:args.limit], frames[:args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.modes:
        mode = Mode(name)
        poses, timing, _ = run(frames, mode)
        save_tum(out / f"traj_{mode.value}.tum", stamps, poses)
        np.savetxt(out / f"timing_{mode.value}.csv", timing, delimiter=",",
                   header="preproc,fft,icp,map,icp_iters,keyframe", comments="")
        ms = timing[1:, :4].sum(axis=1) * 1e3
        print(f"{mode.value:8s} frames {len(poses)}  ms/frame p50 {np.median(ms):.1f} "
              f"p95 {np.percentile(ms, 95):.1f}  fft {timing[1:,1].mean()*1e3:.2f} "
              f"icp {timing[1:,2].mean()*1e3:.2f} map {timing[1:,3].mean()*1e3:.2f}")


if __name__ == "__main__":
    main()
