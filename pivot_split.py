"""PS (Sect. 5.1): the pivot split of Castellaz et al., planar, written from the papers.

Sources:
  [T25] Castellaz, Wouda, van Beijnum, IEEE TNSRE 33 (2025) 4277-4287, doi 10.1109/TNSRE.2025.3622734
  [M24] Castellaz et al., IEEE MeMeA 2024, doi 10.1109/MeMeA60663.2024.10596802 (the split method and the pivot)
Method, per frame (x forward, z up; the insole gives only the CoP and the contact timing, never a force magnitude):
  1. total force: the whole-body force sums (Sx, Sz) of the kinematics, the same totals the graph receives, smoothed by
     a centered moving mean (MATLAB movmean, 'shrink' ends): horizontal 0.1 s, vertical 0.05 s [T25 II-E].
     Option how="2024": zero-phase Butterworth instead, horizontal band-pass 0.1-5 Hz (order 4), vertical low-pass 6 Hz
     (order 8), and no smoothing after the split [M24 II-B].
  2. pivot = pelvis point + (0, 0.5 BH), along the vertical, the same for every subject [M24 II-C; T25 II-F].
  3. direction of each foot force: unit vector from its CoP (height cop_z, default 0) to the pivot.
  4. double support: magnitudes w >= 0 by least squares on [Fx; Fz] = [vR vL] w (two-variable NNLS by active-set
     enumeration, as lsqnonneg); single support: the whole total force to the foot in contact [T25 II-F]; neither
     foot: 0.
  5. Savitzky-Golay filter, order 1, 15 frames, on the full series of every foot and component [T25 II-F].
Contact: insole Fz > f_contact. No force plate value is used.
"""
import numpy as np
from scipy.signal import butter, savgol_filter, sosfiltfilt

VPP_BH = 0.5

def movmean(x, k):
    """MATLAB movmean(x, k), default 'shrink' ends: window [t - k//2, t + (k - 1)//2]."""
    x = np.asarray(x, float)
    n = len(x)
    c = np.concatenate([[0.0], np.cumsum(x)])
    t = np.arange(n)
    lo = np.maximum(t - k // 2, 0)
    hi = np.minimum(t + (k - 1) // 2, n - 1) + 1
    return (c[hi] - c[lo]) / (hi - lo)


def total_force(Sx, Sz, fs, how="2025", ap_s=0.1, v_s=0.05):
    """Step 1: the smoothed total force."""
    Sx, Sz = np.asarray(Sx, float), np.asarray(Sz, float)
    if how == "2025":
        return movmean(Sx, int(round(ap_s * fs))), movmean(Sz, int(round(v_s * fs)))
    if how == "2024":
        bp = butter(4, [0.1, 5.0], btype="bandpass", fs=fs, output="sos")
        lp = butter(8, 6.0, btype="lowpass", fs=fs, output="sos")
        return sosfiltfilt(bp, Sx), sosfiltfilt(lp, Sz)
    if how == "none":
        return Sx, Sz
    raise ValueError(how)


def nnls2(vR, vL, F):
    """Exact two-variable NNLS per frame: min |w_R vR + w_L vL - F| over w >= 0 (vR, vL unit, (n, 2))."""
    n = len(F)
    D = vR[:, 0] * vL[:, 1] - vR[:, 1] * vL[:, 0]
    ok = np.abs(D) > 1e-9
    Ds = np.where(ok, D, 1.0)
    wR = (F[:, 0] * vL[:, 1] - F[:, 1] * vL[:, 0]) / Ds
    wL = (F[:, 1] * vR[:, 0] - F[:, 0] * vR[:, 1]) / Ds
    # degenerate (parallel directions): minimum-norm least squares (pseudo-inverse)
    pR = np.sum(vR * F, axis=1)
    wR = np.where(ok, wR, 0.5 * pR)
    wL = np.where(ok, wL, 0.5 * pR)
    # active sets: both free, only R, only L, none; the feasible one with the least residual (first on ties)
    W = np.stack([np.column_stack([wR, wL]),
                  np.column_stack([np.maximum(pR, 0.0), np.zeros(n)]),
                  np.column_stack([np.zeros(n), np.maximum(np.sum(vL * F, axis=1), 0.0)]),
                  np.zeros((n, 2))], axis=1)
    r = W[:, :, [0]] * vR[:, None, :] + W[:, :, [1]] * vL[:, None, :] - F[:, None, :]
    cost = np.where(np.all(W >= 0, axis=2), np.sum(r * r, axis=2), np.inf)
    return W[np.arange(n), np.argmin(cost, axis=1)]


def split(Fx, Fz, copxR, copxL, vpp, onR, onL, cop_z=0.0):
    """Steps 3-4. vpp (n, 2); cop_z the CoP height (m). Returns {side: (Fx, Fz)} before the output smoothing."""
    Fx, Fz = np.asarray(Fx, float), np.asarray(Fz, float)
    n = len(Fx)
    onR, onL = np.asarray(onR, bool), np.asarray(onL, bool)
    v = {}
    for s, cx in (("R", copxR), ("L", copxL)):
        d = np.asarray(vpp, float) - np.column_stack([np.asarray(cx, float), np.full(n, cop_z)])
        v[s] = d / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
    out = {s: np.zeros((n, 2)) for s in ("R", "L")}
    ds = onR & onL
    if ds.any():
        w = nnls2(v["R"][ds], v["L"][ds], np.column_stack([Fx[ds], Fz[ds]]))
        out["R"][ds] = w[:, [0]] * v["R"][ds]
        out["L"][ds] = w[:, [1]] * v["L"][ds]
    for s, a, b in (("R", onR, onL), ("L", onL, onR)):
        ss = a & ~b
        out[s][ss, 0], out[s][ss, 1] = Fx[ss], Fz[ss]
    return {s: (out[s][:, 0], out[s][:, 1]) for s in ("R", "L")}


def castellaz(Sx, Sz, pelvis, height, insole, fs, how="2025", ap_s=0.1, vpp_bh=VPP_BH, f_contact=20.0, sg=15,
              cop_z=0.0):
    """The whole method. pelvis: the VPP base point (n, 2); insole: {side: dict(Fy, copx)} (lab CoP x; Fy only for
    contact). Returns {side: (Fx, Fz)} and the contact masks."""
    for a in (Sx, Sz, pelvis, insole["R"]["Fy"], insole["L"]["Fy"], insole["R"]["copx"], insole["L"]["copx"]):
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite input (movmean / Savitzky-Golay would spread it)")
    Fx, Fz = total_force(Sx, Sz, fs, how, ap_s)
    vpp = np.asarray(pelvis, float) + np.array([0.0, vpp_bh * height])
    onR, onL = (np.asarray(insole[s]["Fy"], float) > f_contact for s in ("R", "L"))
    f = split(Fx, Fz, insole["R"]["copx"], insole["L"]["copx"], vpp, onR, onL, cop_z)
    if how == "2025" and sg:
        f = {s: tuple(savgol_filter(c, sg, 1, mode="interp") for c in f[s]) for s in f}
    return f, (onR, onL)
