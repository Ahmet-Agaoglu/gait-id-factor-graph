"""Effect of the hip-center rule (Sect. 3). The camera model puts the hip centers 0.048 body heights below the mean of
RASI, LASI and SACR (sacrum counted twice, wang_segments.py). Harrington et al. (2007) also give an anterior-posterior
term, -(0.24 PD + 9.9 mm) from the ASIS midpoint along the pelvis (PD: pelvic depth, here ASIS midpoint to sacrum in the
sagittal plane). The centroid lies 0.5 PD behind the ASIS midpoint, so the rule puts the hips (0.26 PD - 9.9 mm) further
back along the pelvis. For every trial: PD, that shift and its horizontal part, and, with the hips moved forward by it
(vertical rule unchanged), the change of the whole-body CoM x_G and of the constant moment-balance term -dx_G S_z / W over
the insole's contact frames (m BW). Markers and insoles only, no force plate channel.
Output: results/hip_center/
Usage: python -I hip_center.py run [workers]
"""
import datetime
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import factor_graph as L  # noqa: E402
import camera_mc as K  # noqa: E402
import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "hip_center")
F_CSV, F_TXT = os.path.join(OUT, "hip_center.csv"), os.path.join(OUT, "hip_center_summary.txt")


def one(subj, trial):
    import planar_model as sp_
    import wang_segments as wr
    R = K.Wang(subj, trial)
    n, FS = R.n, R.FS
    P, pelvis, _ = wr.planar_from_markers((None, R.raw), n)
    P = {k: (sp_.lowpass(v, 6.0, FS) if isinstance(v, np.ndarray) else v) for k, v in P.items()}
    pelvis = sp_.lowpass(pelvis, 6.0, FS)
    asis = 0.5 * (P["R.ASIS"] + P["L.ASIS"])
    sacr = P["R.PSIS"]
    v = asis - sacr
    pd_ = np.linalg.norm(v, axis=1)
    u = v / pd_[:, None]                                  # anterior along the pelvis
    shift = (0.26 * pd_ - 0.0099)[:, None] * u            # forward, from our centroid to Harrington's AP position
    hip0 = wr.hip_from_pelvis_planar(pelvis, R.height)
    hip1 = hip0.copy()
    hip1[:, 0] += shift[:, 0]                             # horizontal part only; vertical rule unchanged
    P0 = dict(P, **{"R.Hip": hip0, "L.Hip": hip0})
    P1 = dict(P, **{"R.Hip": hip1, "L.Hip": hip1})
    s0, s1 = (wr.build(Q, R.mass, R.height, n) for Q in (P0, P1))
    k0, k1 = L.whole_body_terms(s0), L.whole_body_terms(s1)
    _, ins, BW, inf = LI.load(subj, trial, "camera")
    m = (ins["R"]["Fy"] > L.INSOLE["F_CONTACT"]) | (ins["L"]["Fy"] > L.INSOLE["F_CONTACT"])
    m = m & inf["win"]
    k = len(m)
    dxg = k1["xc"][:k] - k0["xc"][:k]
    chk = float(np.max(np.abs(k0["xc"][:k] - L.whole_body_terms(R.segs0)["xc"][:k])))
    return dict(subject=subj, trial=trial, pd_mm=1000 * float(pd_.mean()), shift_mm=1000 * float(np.linalg.norm(shift, axis=1).mean()),
                shift_x_mm=1000 * float(shift[:, 0].mean()), dxG_mm=1000 * float(dxg[m].mean()),
                moment_mBW=float(np.mean(-dxg[m] * k0["Sz"][:k][m]) / BW), check_xc_m=chk)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    d = pd.DataFrame([one(s, t) for s, t in RW.trials()])
    d.to_csv(F_CSV, index=False)
    S = d.groupby("subject").mean(numeric_only=True)
    lines = [f"hip_center.py, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(d)} trials, {len(S)} subjects; "
             f"check: rebuilt x_G vs segs0 max |diff| {d.check_xc_m.max():.1e} m.",
             "Per subject (mean over trials), then median over trials. shift = hips moved forward to Harrington's AP position; "
             "moment = -dx_G S_z / W, mean over the insole's contact frames (m BW)."]
    lines.append(S[["pd_mm", "shift_mm", "shift_x_mm", "dxG_mm", "moment_mBW"]].round(4).to_string())
    for c in ("pd_mm", "shift_mm", "shift_x_mm", "dxG_mm", "moment_mBW"):
        lines.append(f"{c}: median over trials {d[c].median():.4f}, subject means {S[c].min():.4f} to {S[c].max():.4f}")
    txt = "\n".join(lines)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
