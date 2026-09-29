"""Synthetic flight: known trajectory through a random courtyard.
Each mode must track it; a deliberately broken sign must not."""
import sys

import numpy as np
from scipy.spatial.transform import Rotation

import odom

rng = np.random.default_rng(1)


def world():
    parts = [np.c_[rng.uniform(-40, 40, (60000, 2)), np.zeros(60000)]]       # ground
    for _ in range(30):                                                       # walls
        p0 = rng.uniform(-35, 35, 2)
        d = odom.rot2(rng.uniform(0, np.pi)) @ [1.0, 0.0]
        n = 3000
        t = rng.uniform(0, rng.uniform(4, 15), n)
        h = rng.uniform(0, rng.uniform(3, 12), n)
        parts.append(np.c_[p0 + t[:, None] * d, h])
    for _ in range(25):                                                       # pillars
        c = rng.uniform(-35, 35, 2)
        a = rng.uniform(0, 2 * np.pi, 800)
        parts.append(np.c_[c + 0.4 * np.c_[np.cos(a), np.sin(a)], rng.uniform(0, 8, 800)])
    return np.concatenate(parts)


def trajectory(n=150):
    t = np.linspace(0, 1, n)
    pos = np.c_[12 * np.sin(2 * np.pi * t), 8 * np.sin(4 * np.pi * t), 3 + 1.5 * t]
    yaw = 1.2 * np.sin(2 * np.pi * t)
    rp = 0.05 * np.c_[np.sin(6 * np.pi * t), np.cos(5 * np.pi * t)]
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_euler("xyz", np.c_[rp, yaw]).as_matrix()
    T[:, :3, 3] = pos
    return T


def scan(W, T):
    p = odom.apply(np.linalg.inv(T), W)
    r = np.linalg.norm(p, axis=1)
    elev_h = np.degrees(np.arcsin(p[:, 2] / r))              # horizontal lidar
    elev_v = np.degrees(np.arcsin(p[:, 1] / r))              # vertical lidar, spins in XZ
    m = (r > 1) & (r < 35) & ((np.abs(elev_h) < 16.6) | (np.abs(elev_v) < 16.6))
    p = p[m]
    p = p[rng.random(len(p)) < 0.35]
    return p + rng.normal(0, 0.02, p.shape)


def ate(est, gt):
    """Frame-0 aligned position error (both start at identity)."""
    rel = np.linalg.inv(gt[0]) @ gt
    return np.linalg.norm(est[:, :3, 3] - rel[:, :3, 3], axis=1)



if __name__ == "__main__":
    W = world()
    stride = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    gt = trajectory(300)[::stride]
    frames = [scan(W, T) for T in gt]
    print(f"{len(frames)} frames, ~{np.mean([len(f) for f in frames]):.0f} pts each, "
          f"step {np.linalg.norm(np.diff(gt[:, :3, 3], axis=0), axis=1).mean():.2f} m")

    for mode in odom.Mode:
        poses, timing, _ = odom.run(frames, mode)
        e = ate(poses, gt)
        ms = timing[1:, :4].sum(axis=1) * 1e3
        print(f"{mode.value:8s} ATE mean {e.mean():.3f} max {e.max():.3f} m | "
              f"p50 {np.median(ms):.1f} ms  fft {timing[1:,1].mean()*1e3:.1f}  icp {timing[1:,2].mean()*1e3:.1f}")

    if stride > 1:
        sys.exit()

    # Comparator must be able to fail: flip the FFT yaw sign and expect drift
    orig = odom.plane_rot
    odom.plane_rot = lambda axes, a: orig(axes, -a)
    poses, _, _ = odom.run(frames, odom.Mode.FFT)
    print(f"broken-sign fft ATE mean {ate(poses, gt).mean():.3f} m (should be much worse)")
