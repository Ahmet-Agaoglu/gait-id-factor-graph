"""Insole contact threshold, measured plate-free (Sect. 4.1). Kinematic gait events from the camera markers (Zeni et al.
2008: heel strike = most anterior heel relative to the hip, toe-off = most posterior MT5 relative to the hip; sagittal,
walking toward +x). For each threshold, the insole contact onsets and ends are matched to the nearest heel strike and toe-off
of the same foot; timing error RMS per foot and trial. Chosen threshold: the smallest median RMS over foot-trials (grid
20-150 N, 10 N). No force plate channel is read.
Output: results/contact_threshold/
Usage: python -I contact_threshold.py run [workers]
"""
import datetime
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402

import camera_mc as K  # noqa: E402

THR = np.arange(20, 160, 10)
OUT = os.path.join(config.RESULTS, "contact_threshold")


def one(args):
    subj, trial = args
    R = K.Wang(subj, trial)
    P, fs, n = R.P0, R.FS, R.n
    ms = 1000.0 / fs
    rows = []
    for s in ("R", "L"):
        hip = P[s + ".Hip"][:n, 0]
        hs = find_peaks(P[s + ".Heel"][:n, 0] - hip, distance=int(0.5 * fs), prominence=0.05)[0]
        to = find_peaks(-(P[s + ".MT5"][:n, 0] - hip), distance=int(0.5 * fs), prominence=0.05)[0]
        f = R.ins[s]["Fy"][:n]
        for thr in THR:
            on = f > thr
            up = np.where(~on[:-1] & on[1:])[0] + 1
            dn = np.where(on[:-1] & ~on[1:])[0] + 1
            don = np.array([ms * (up[np.argmin(np.abs(up - h))] - h) for h in hs]) if len(up) else np.array([np.nan])
            doff = np.array([ms * (dn[np.argmin(np.abs(dn - t))] - t) for t in to]) if len(dn) else np.array([np.nan])
            rows.append(dict(subject=subj, trial=trial, side=s, thr=int(thr), n_hs=len(hs), n_to=len(to),
                             on_ms=float(np.median(don)), off_ms=float(np.median(doff)),
                             rms_ms=float(np.sqrt(np.mean(np.r_[don, doff] ** 2))), contacts_per_stride=len(up) / max(len(hs), 1),
                             contact_share=float(on[R.win].mean()), ds_share=float(((R.ins["R"]["Fy"] > thr) &
                                                                                   (R.ins["L"]["Fy"] > thr))[R.win].mean())))
    return rows


def run(workers=4):
    import trials as TR
    os.makedirs(OUT, exist_ok=True)
    f_csv, f_txt = os.path.join(OUT, "contact_threshold.csv"), os.path.join(OUT, "contact_threshold_summary.txt")
    for f in (f_csv, f_txt):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    J = TR.trials()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        d = pd.DataFrame([r for rr in pool.map(one, J) for r in rr])
    d.to_csv(f_csv, index=False)
    m = d.groupby("thr")[["on_ms", "off_ms", "rms_ms", "contacts_per_stride", "contact_share", "ds_share"]].median()
    best = int(m.rms_ms.idxmin())
    txt = "\n".join([f"contact_threshold.py, {datetime.datetime.now().isoformat(timespec='seconds')}: "
                     f"{d.subject.nunique()} subjects, {d.groupby(['subject', 'trial']).ngroups} trials, both feet, "
                     f"{time.time() - t0:.0f} s. Medians over foot-trials.", m.round(3).to_string(),
                     f"CHOSEN threshold (smallest median timing RMS): {best} N"])
    print(txt)
    with open(f_txt, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
