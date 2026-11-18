"""Planar (sagittal) body model: segment table (Winter 2009), filter, derivatives and the bottom-up inverse dynamics of one
leg. Frame: x forward, y up. Units: m, N, kg, s.
"""
import numpy as np
from scipy.signal import butter, filtfilt

MASS = 74.3      # kg; set per trial by the scripts
HEIGHT = 1.725   # m; set per trial by the scripts
G = 9.81
FS = 150.0       # Hz; set per trial by the scripts (wang_segments.build: 100 Hz)
F_CONTACT = 20.0  # N; below it inverse_dynamics applies no CoP (the force is zero there)

# Winter (2009) anthropometric table: mass fraction, COM fraction from proximal, rg/L about COM
SEG = {
    "foot":  dict(m=0.0145, c=0.50, rg=0.475),
    "shank": dict(m=0.0465, c=0.433, rg=0.302),
    "thigh": dict(m=0.100, c=0.433, rg=0.323),
    "hat":   dict(m=0.678, c=0.626, rg=0.496),
}


def lowpass(x, fc, fs, order=4):
    b, a = butter(order, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=0)


def deriv(x, fs):
    v = np.gradient(x, 1.0 / fs, axis=0)
    a = np.gradient(v, 1.0 / fs, axis=0)
    return v, a


def cross2(r, F):
    """z-component of r x F for (N,2) arrays."""
    return r[:, 0] * F[:, 1] - r[:, 1] * F[:, 0]


def inverse_dynamics(segs, side, Fx, Fy, copx, copy=None):
    """Bottom-up planar Newton-Euler for one leg. Returns moments (N m) at ankle, knee, hip:
    moment applied by the proximal segment on the distal segment, +z (counter-clockwise, X->Y).
    copy: height of the centre of pressure; None (level ground) means y = 0."""
    gvec = np.array([0.0, -G])
    foot, shank, thigh = segs[(side, "foot")], segs[(side, "shank")], segs[(side, "thigh")]
    n = len(Fx)
    F_grf = np.column_stack([Fx, Fy])
    on = Fy > F_CONTACT
    cy = np.zeros(n) if copy is None else np.where(on, copy, 0.0)
    r_cop = np.column_stack([np.where(on, copx, foot["com"][:, 0]), cy])
    # foot
    F_ank = foot["m"] * foot["acom"] - foot["m"] * gvec - F_grf
    M_ank = (foot["I"] * foot["alpha"]
             - cross2(r_cop - foot["com"], F_grf)
             - cross2(foot["prox"] - foot["com"], F_ank))
    # shank (distal joint = ankle carries -F_ank, -M_ank)
    F_knee = shank["m"] * shank["acom"] - shank["m"] * gvec + F_ank
    M_knee = (shank["I"] * shank["alpha"] + M_ank
              - cross2(shank["prox"] - shank["com"], F_knee)
              - cross2(shank["dist"] - shank["com"], -F_ank))
    # thigh
    F_hip = thigh["m"] * thigh["acom"] - thigh["m"] * gvec + F_knee
    M_hip = (thigh["I"] * thigh["alpha"] + M_knee
             - cross2(thigh["prox"] - thigh["com"], F_hip)
             - cross2(thigh["dist"] - thigh["com"], -F_knee))
    return dict(ankle=M_ank, knee=M_knee, hip=M_hip)


def newton_baseline(S, FyR, FyL):
    onR, onL = FyR > F_CONTACT, FyL > F_CONTACT
    tot = np.where(onR, FyR, 0.0) + np.where(onL, FyL, 0.0)
    tot[tot < 1e-6] = np.nan
    fR = np.where(onR, S * np.where(onR, FyR, 0.0) / tot, 0.0)
    fL = np.where(onL, S * np.where(onL, FyL, 0.0) / tot, 0.0)
    return np.nan_to_num(fR), np.nan_to_num(fL)
