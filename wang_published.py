"""WP (Sect. 5.1): the wearable inverse dynamics released with the data set (ID_Portable.sto, OpenSim inverse dynamics
from the IMU kinematics and the insoles), scored against the planar reference solution over the same window.
Its insole force file has no horizontal force, so its shear error is that of ZS.
"""
import numpy as np

JOINTS = ("ankle", "knee", "hip")
REF_SIGN = {"ankle": 1.0, "knee": 1.0, "hip": 1.0}   # released moments have the sign of the planar reference


def published(T, row_zero):
    """Wang et al.'s released wearable inverse dynamics (ID_Portable.sto) against the planar
    reference solution over the scoring window T["trim"]."""
    d, n, mass, trim = T["d"], T["n"], T["mass"], T["trim"]
    cols = dict(ankle="ankle_angle_r_moment", knee="knee_angle_r_moment", hip="hip_flexion_r_moment")
    m = min(n, len(d["id_lab"]), len(d["id_wear"]))
    w = trim[:m]
    out = {f"Wang_published_{k}": row_zero[f"zero_shear_{k}"]
           for k in ("stance", "ds", "ss", "vert_stance", "vert_ds", "vert_ss")}
    for j in JOINTS:
        x = REF_SIGN[j] * d["id_wear"][cols[j]].values[:m][w] / mass
        y = T["ref_m"][j][:m][w] / mass
        ok = np.isfinite(x) & np.isfinite(y)
        if T["lab_ok"] and ok.sum() >= 10:
            out[f"Wang_published_{j}"] = float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))
            out[f"Wang_published_{j}_r"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
        else:
            out[f"Wang_published_{j}"] = out[f"Wang_published_{j}_r"] = np.nan
    out["Wang_published_n_frames"] = int(m)
    return out
