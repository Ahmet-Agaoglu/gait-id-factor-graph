"""Planar segment model of Wang et al. (2023) from joint points: marker reading, hip joint centers, the seven-segment
table and the placement of the insole CoP on the foot.
"""
import numpy as np
import pandas as pd

import planar_model as sp_

FS = 100.0

def planar_from_markers(trc, n, max_gap=0.05):
    """Planar positions from the lab markers, in the OpenSim/XLD frame (x forward, y up).
    Short marker gaps are interpolated; the flag reports whether every required marker stayed
    below max_gap missing samples."""
    t, M = trc
    state = {"ok": True}

    def xy(k):
        a = np.column_stack([M[k][:n, 0], M[k][:n, 1]])
        bad = np.isnan(a[:, 0])
        if bad.any():
            if bad.mean() > max_gap:
                state["ok"] = False
            a = pd.DataFrame(a).interpolate(limit_direction="both").values
        return a
    P = {}
    for side, (kn, an, to, he) in (("R", ("RLFE", "RLM", "R5MT", "RCAL")),
                                   ("L", ("LLFE", "LLM", "L5MT", "LCAL"))):
        P[side + ".Knee"], P[side + ".Ankle"] = xy(kn), xy(an)
        P[side + ".MT5"], P[side + ".Heel"] = xy(to), xy(he)
    # pelvis landmarks: ASIS pair + sacrum (no PSIS markers in this set)
    P["R.ASIS"], P["L.ASIS"] = xy("RASI"), xy("LASI")
    P["R.PSIS"] = P["L.PSIS"] = xy("SACR")
    pelvis = 0.25 * (P["R.ASIS"] + P["L.ASIS"] + P["R.PSIS"] + P["L.PSIS"])
    # hip joint centres: Harrington-like offset below/behind the pelvis centroid is not available
    # in 2-D from these markers; use the ASIS/sacrum centroid shifted down by 9 % of height later.
    P["pelvis_pts"] = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]
    return P, pelvis, state["ok"]


def hip_from_pelvis_planar(pelvis, height):
    """Planar hip joint centre below the pelvis landmark centroid. Harrington (2007):
    vertical offset = -(0.30 PW + 10.9 mm); with PW ~ 0.138 height this is ~ 0.048 height."""
    out = pelvis.copy()
    out[:, 1] -= 0.048 * height
    return out


def build(P_joints, mass, height, n):
    """Build the 7-segment planar model dict used by planar_model.inverse_dynamics."""
    sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, FS
    segs = {}
    for s in ("R", "L"):
        hip, knee, ank, toe = (P_joints[s + k][:n] for k in (".Hip", ".Knee", ".Ankle", ".MT5"))
        for name, prox, dist in (("foot", ank, toe), ("shank", knee, ank), ("thigh", hip, knee)):
            par = sp_.SEG[name]
            L = np.linalg.norm(dist - prox, axis=1).mean()
            m = par["m"] * mass
            com = prox + par["c"] * (dist - prox)
            _, acom = sp_.deriv(com, FS)
            th = np.unwrap(np.arctan2((dist - prox)[:, 1], (dist - prox)[:, 0]))
            _, alpha = sp_.deriv(th, FS)
            segs[(s, name)] = dict(m=m, I=m * (par["rg"] * L) ** 2, com=com, acom=acom,
                                   alpha=alpha, prox=prox, dist=dist, L=L)
    midhip = 0.5 * (P_joints["R.Hip"][:n] + P_joints["L.Hip"][:n])
    L_hat = 0.288 * height
    com = midhip + np.array([0.0, sp_.SEG["hat"]["c"] * L_hat])
    _, acom = sp_.deriv(com, FS)
    m = sp_.SEG["hat"]["m"] * mass
    segs[("B", "hat")] = dict(m=m, I=m * (sp_.SEG["hat"]["rg"] * L_hat) ** 2, com=com, acom=acom,
                              alpha=np.zeros(n), prox=midhip, dist=com, L=L_hat)
    return segs


def insole_cop_to_lab(P_joints, side, cop_local, n, heel_key=None):
    """The insole COP is expressed in the foot (calcaneus) frame: a distance from the heel end
    along the foot. Place it in the lab frame on the heel->toe line.
    heel_key: a measured heel marker, else the heel is estimated from ankle and toe."""
    ank, toe = P_joints[side + ".Ankle"][:n], P_joints[side + ".MT5"][:n]
    if heel_key is not None and heel_key in P_joints:
        heel = P_joints[heel_key][:n]
    else:                                   # ankle sits ~25 % of foot length ahead of the heel
        v = toe - ank
        L = np.linalg.norm(v, axis=1, keepdims=True)
        u = v / np.maximum(L, 1e-9)
        heel = ank - 0.25 * (L / 0.75) * u
    v = toe - heel
    u = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)
    return heel[:, 0] + cop_local[:n] * u[:, 0]
