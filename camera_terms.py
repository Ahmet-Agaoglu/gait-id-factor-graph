"""Camera kinematics of one trial: the planar segment table from the raw markers and its whole-body terms.

terms() is the camera input of the graph and of every competitor: markers through the 6 Hz filter, hip joint centers
below the pelvis markers (wang_segments.hip_from_pelvis_planar), the planar segment model (wang_segments.build) and the
whole-body force sums Sx, Sz and the rate of change of the angular momentum about the CoM, dH/dt.
hat_on_line() replaces the rigid HAT by a HAT on a measured line (camera_mc.py, source 'meas').
"""
import numpy as np

import planar_model as sp_

FS = 100.0


def com_terms(segs):
    """Whole-body COM position, Sx, Sz and dH/dt about the COM from the segment table."""
    M = sum(v["m"] for v in segs.values())
    rc = sum(v["m"] * v["com"] for v in segs.values()) / M
    ac = sum(v["m"] * v["acom"] for v in segs.values()) / M
    Hd = sum(v["I"] * v["alpha"] + v["m"] * sp_.cross2(v["com"] - rc, v["acom"] - ac) for v in segs.values())
    Sx = sum(v["m"] * v["acom"][:, 0] for v in segs.values())
    Sz = sum(v["m"] * (v["acom"][:, 1] + sp_.G) for v in segs.values())
    return rc, Hd, Sx, Sz


def terms(M, n, mass, height, seg=None):
    """Sx, Sz, dH/dt of the rigid-HAT camera model from the marker dict M; also the segment table and the joint points.
    seg: a segment parameter table in place of planar_model.SEG (camera_mc.py)."""
    import wang_segments as wr
    P, pelvis, _ = wr.planar_from_markers((None, M), n)
    P = {k: (sp_.lowpass(v, 6.0, FS) if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    pelvis = sp_.lowpass(pelvis, 6.0, FS)
    P["R.Hip"] = P["L.Hip"] = wr.hip_from_pelvis_planar(pelvis, height)
    old = sp_.SEG
    if seg is not None:
        sp_.SEG = seg
    try:
        segs = wr.build(P, mass, height, n)
    finally:
        sp_.SEG = old
    rc, Hd, Sx, Sz = com_terms(segs)
    return dict(Sx=Sx, Sz=Sz, Hd=Hd), segs, P


def hat_on_line(segs, hip, top, mass):
    """Replace the HAT entry by a segment on the hip -> top line."""
    par = sp_.SEG["hat"]
    v = top - hip
    L = float(np.linalg.norm(v, axis=1).mean())
    com = hip + par["c"] * v
    _, acom = sp_.deriv(com, sp_.FS)
    th = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    _, alpha = sp_.deriv(th, sp_.FS)
    m = par["m"] * mass
    out = dict(segs)
    out[("B", "hat")] = dict(m=m, I=m * (par["rg"] * L) ** 2, com=com, acom=acom, alpha=alpha, prox=hip, dist=top,
                             L=L, theta=th)
    return out
