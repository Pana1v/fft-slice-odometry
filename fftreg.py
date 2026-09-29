"""FFT slice registration: rasterize 3D points into 2D slice images, then
recover in-plane rotation (Fourier-Mellin) and shift (phase correlation).

Convention everywhere: a result s means  mov(x) = ref(x - s),
i.e. mov is ref moved by +s. To align mov onto ref, apply -s.
"""
import numpy as np
from scipy import fft as sfft
from scipy.ndimage import map_coordinates

N_PIX = 256
RES = 0.25          # m per pixel -> 64 m field of view
BLUR_PX = 1.0       # 16-beam scans are sparse rings; applied as a spectral weight, not a spatial filter
N_ANG = 360         # over [0, pi) -> 0.5 deg per bin before subpixel
N_RAD = 96
FFT_WORKERS = 4
# Peak search windows around the prediction. Rectilinear scenes alias under
# 90 deg rotations and repeated facades alias in shift; at 10 Hz the true
# correction stays well inside these bounds.
MAX_SHIFT_M = 8.0
MAX_ROT = np.deg2rad(20.0)


def _parabolic(ym, y0, yp):
    """Subpixel offset of a peak from its two neighbours."""
    den = ym - 2.0 * y0 + yp
    if abs(den) < 1e-12:
        return 0.0
    return 0.5 * (ym - yp) / den


class SliceGrid:
    """Precomputed windows and log-polar sampling for one image size."""

    def __init__(self, n=N_PIX, res=RES):
        self.n = n
        self.res = res
        self.half = n * res / 2.0

        w = np.hanning(n)
        self.hann = np.outer(w, w).astype(np.float32)

        # Real input -> half spectrum: axis 0 is kx (full), axis 1 is ky >= 0
        kx = sfft.fftfreq(n)[:, None]
        ky = sfft.rfftfreq(n)[None, :]
        kk = np.hypot(kx, ky)

        # Low-pass on the cross-power: pure phase correlation whitens noise too
        self.lowpass = np.exp(-(kk / 0.25) ** 2).astype(np.float32)

        # Band-pass on the magnitude before log-polar: DC carries no angle info,
        # and the Gaussian stands in for the blur a spatial filter would add
        ks = sfft.fftshift(kk, axes=0)
        self.highpass = (ks * np.exp(-2.0 * (np.pi * BLUR_PX * ks) ** 2)).astype(np.float32)

        # The ky >= 0 half-plane is exactly theta in [0, pi), all rotation needs
        c = n / 2.0
        theta = np.linspace(0.0, np.pi, N_ANG, endpoint=False)
        rad = np.exp(np.linspace(np.log(3.0), np.log(c - 2.0), N_RAD))
        self.lp_coords = np.stack([
            c + rad[:, None] * np.cos(theta)[None, :],
            rad[:, None] * np.sin(theta)[None, :],
        ])
        self.rad_win = np.hanning(N_RAD)[:, None]

        wrapped = (np.arange(n) + n // 2) % n - n // 2
        near = np.abs(wrapped) * res <= MAX_SHIFT_M
        self.shift_win = near[:, None] & near[None, :]
        k = (np.arange(N_ANG) + N_ANG // 2) % N_ANG - N_ANG // 2
        self.rot_win = np.abs(k) * np.pi / N_ANG <= MAX_ROT

    # ---------- rasterize ----------

    def render(self, uv, band=None, n_bands=1):
        """(N,2) metric coords centred on the grid -> (n_bands, n, n) windowed images.
        band gives each point's band index; one bincount covers all bands."""
        ij = np.floor((uv + self.half) / self.res).astype(np.int64)
        keep = (ij[:, 0] >= 0) & (ij[:, 0] < self.n) & (ij[:, 1] >= 0) & (ij[:, 1] < self.n)
        if band is not None:
            keep &= (band >= 0) & (band < n_bands)
        ij = ij[keep]
        flat = ij[:, 0] * self.n + ij[:, 1]
        if band is not None:
            flat += band[keep] * self.n * self.n
        img = np.bincount(flat, minlength=n_bands * self.n * self.n)
        img = np.log1p(img.reshape(n_bands, self.n, self.n), dtype=np.float32)

        # log1p so dense map cells do not drown out the sparse scan cells.
        # No spatial blur: normalized phase correlation would cancel it anyway.
        img *= self.hann
        return img

    def spectra(self, imgs):
        return sfft.rfft2(imgs, workers=FFT_WORKERS)

    # ---------- translation ----------

    def _peak(self, corr):
        i, j = np.unravel_index(np.argmax(np.where(self.shift_win, corr, -np.inf)), corr.shape)
        n = self.n
        di = _parabolic(corr[(i - 1) % n, j], corr[i, j], corr[(i + 1) % n, j])
        dj = _parabolic(corr[i, (j - 1) % n], corr[i, j], corr[i, (j + 1) % n])

        # Wrap to signed shift
        si = (i + di + n / 2) % n - n / 2
        sj = (j + dj + n / 2) % n - n / 2
        return np.array([si, sj]) * self.res, float(corr[i, j])

    def _corr(self, cross):
        cross = cross / (np.abs(cross) + 1e-9)
        return sfft.irfft2(cross * self.lowpass, s=(self.n, self.n), workers=FFT_WORKERS)

    def phase_corr(self, F_ref, F_mov):
        """Multi-channel phase correlation. Returns (shift_m (2,), peak, corr)."""
        corr = self._corr((F_mov * np.conj(F_ref)).sum(axis=0))
        s, peak = self._peak(corr)
        return s, peak, corr

    def phase_corr_each(self, F_ref, F_mov):
        """Independent phase correlation per channel, batched. Returns (shifts (B,2), peaks (B,))."""
        corr = self._corr(F_mov * np.conj(F_ref))
        out = [self._peak(c) for c in corr]
        return np.array([o[0] for o in out]), np.array([o[1] for o in out])

    # ---------- rotation ----------

    def _logpolar(self, F):
        mag = sfft.fftshift(np.abs(F).sum(axis=0), axes=0) * self.highpass
        lp = map_coordinates(mag, self.lp_coords, order=1)
        return lp * self.rad_win

    def lp_spectrum(self, F):
        """Angular spectrum of the log-polar magnitude; cacheable for the map."""
        return sfft.rfft(self._logpolar(F), axis=1, workers=FFT_WORKERS)

    def rot_fm(self, F_ref, F_mov, A=None):
        """In-plane rotation (rad) of mov relative to ref, in (-pi/2, pi/2]."""
        if A is None:
            A = self.lp_spectrum(F_ref)
        B = self.lp_spectrum(F_mov)
        cross = (B * np.conj(A)).sum(axis=0)
        cross /= np.abs(cross) + 1e-9
        corr = sfft.irfft(cross, n=N_ANG, workers=FFT_WORKERS)

        k = int(np.argmax(np.where(self.rot_win, corr, -np.inf)))
        dk = _parabolic(corr[(k - 1) % N_ANG], corr[k], corr[(k + 1) % N_ANG])
        s = (k + dk + N_ANG / 2) % N_ANG - N_ANG / 2
        return s * np.pi / N_ANG


def rot2(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


if __name__ == "__main__":
    # Synthetic check on a random "room": known rotation + shift must come back
    rng = np.random.default_rng(0)
    walls = []
    for _ in range(40):
        p0 = rng.uniform(-25, 25, 2)
        d = rot2(rng.uniform(0, np.pi)) @ np.array([1.0, 0.0])
        t = rng.uniform(0, rng.uniform(3, 12), 400)
        walls.append(p0 + t[:, None] * d)
    ref_pts = np.concatenate(walls)

    g = SliceGrid()
    F_ref = g.spectra(g.render(ref_pts))

    def check(yaw_deg, shift):
        mov = ref_pts @ rot2(np.deg2rad(yaw_deg)).T + shift
        mov = mov[rng.random(len(mov)) < 0.3]          # sparser, like a scan
        F_mov = g.spectra(g.render(mov))
        a = g.rot_fm(F_ref, F_mov)
        # derotate, then shift
        F_mov2 = g.spectra(g.render(mov @ rot2(-a).T))
        s, peak, _ = g.phase_corr(F_ref, F_mov2)
        # derotating also rotated the shift vector
        s_true = rot2(-a) @ shift
        return np.rad2deg(a), s, s_true, peak

    for yaw, sh in [(4.0, [1.3, -0.7]), (-11.0, [-3.2, 2.1]), (0.0, [0.26, 0.0]), (18.0, [5.0, 5.0])]:
        a, s, st, pk = check(yaw, np.array(sh))
        ok = abs(a - yaw) < 0.3 and np.linalg.norm(s - st) < 0.05
        print(f"yaw {yaw:6.2f} -> {a:7.3f} deg | shift {st.round(3)} -> {s.round(3)} "
              f"| peak {pk:.3f} | {'OK' if ok else 'FAIL'}")
