"""Isolate the FFT front-end: GT map, perturbed prediction, per-axis residual."""
import time

import numpy as np
from scipy.spatial.transform import Rotation

import odom
import sim_test

W = sim_test.world()
gt = sim_test.trajectory()
frames = [sim_test.scan(W, T) for T in gt]
rng = np.random.default_rng(3)

lmap = odom.LocalMap()
front = odom.FFTFront()
errs, ms = [], []
for i in range(10, len(gt), 3):
    lmap.pts = np.zeros((0, 3))
    for j in range(i - 10, i, 2):
        lmap.update(odom.apply(gt[j], odom.voxel(frames[j], odom.MAP_VOXEL)), gt[j][:3, 3], odom.Mode.FFT_ICP)
    front.prepare(lmap, gt[i - 2][:3, 3])

    d = np.r_[rng.normal(0, np.deg2rad([0.5, 0.5, 3.0])), rng.normal(0, [0.4, 0.4, 0.2])]
    T_pred = odom.exp_se3(d) @ gt[i]
    src = odom.voxel(frames[i], odom.SRC_VOXEL)

    t0 = time.perf_counter()
    T = front.correct(T_pred, src, lmap)
    ms.append((time.perf_counter() - t0) * 1e3)

    E = np.linalg.inv(gt[i]) @ T
    E0 = np.linalg.inv(gt[i]) @ T_pred
    errs.append(np.r_[np.degrees(Rotation.from_matrix(E[:3, :3]).as_euler("xyz")), E[:3, 3],
                      np.degrees(Rotation.from_matrix(E0[:3, :3]).as_euler("xyz")), E0[:3, 3]])

e = np.abs(np.array(errs))
lab = ["roll", "pitch", "yaw", "x", "y", "z"]
print("axis     before  after   (mean abs; deg or m)")
for k, n in enumerate(lab):
    print(f"{n:6s} {e[:, 6 + k].mean():7.3f} {e[:, k].mean():7.3f}")
print(f"fft correct: {np.mean(ms):.1f} ms mean, {len(lmap.pts)} map pts")
