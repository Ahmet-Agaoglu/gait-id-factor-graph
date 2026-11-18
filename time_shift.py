"""Residual time shift between the insoles and the camera kinematics, plate-free (Online Resource 1, Table S2).

Per trial: the total insole Fz of both feet, low-passed at 6 Hz zero-phase like the markers, against the kinematic vertical
force Sz = sum m (a_z + g) from the markers (camera_terms.terms through camera_mc.Wang, the graph's own kinematic input).
Normalized cross-correlation over the scored window (1 s trimmed at each end), lags -300..+300 ms in 10 ms steps, peak
refined by a parabola. Positive lag = the insole is later than the kinematics. The same on the first and the second half of
the window gives the precision of the estimate. No force plate channel is used.
Output: results/time_shift/
Usage: python -I time_shift.py run [workers]
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

import camera_mc as K  # noqa: E402
import factor_graph as F  # noqa: E402
import planar_model as sp_  # noqa: E402

OUT = os.path.join(config.RESULTS, "time_shift")
MAXLAG = 30          # samples at 100 Hz = 300 ms


def lag_of(sz, fz, idx):
    """Lag (samples, sub-sample by a parabola) maximizing corr(sz[t], fz[t + L]) over the frames idx; and the peak r."""
    Ls = np.arange(-MAXLAG, MAXLAG + 1)
    c = np.array([np.corrcoef(sz[idx], fz[idx + L])[0, 1] for L in Ls])
    k = int(np.argmax(c))
    if 0 < k < len(c) - 1:
        den = c[k - 1] - 2 * c[k] + c[k + 1]
        d = 0.5 * (c[k - 1] - c[k + 1]) / den if den < 0 else 0.0
    else:
        d = 0.0
    return float(Ls[k] + d), float(c[k]), bool(0 < k < len(c) - 1)


def one(args):
    subj, trial = args
    R = K.Wang(subj, trial)
    fs = R.FS
    sz = F.whole_body_terms(R.segs0)["Sz"]
    fz = sp_.lowpass(R.ins["R"]["Fy"] + R.ins["L"]["Fy"], 6.0, fs)
    idx = np.where(R.win)[0]
    idx = idx[(idx - MAXLAG >= 0) & (idx + MAXLAG < R.n)]
    h = len(idx) // 2
    L, r, inside = lag_of(sz, fz, idx)
    L1, r1, _ = lag_of(sz, fz, idx[:h])
    L2, r2, _ = lag_of(sz, fz, idx[h:])
    ms = 1000.0 / fs
    return dict(subject=subj, trial=trial, n_win=len(idx), lag_ms=L * ms, r_peak=r, peak_inside=inside,
                lag_first_ms=L1 * ms, lag_second_ms=L2 * ms, r_first=r1, r_second=r2)


def run(workers=10):
    import trials as TR
    os.makedirs(OUT, exist_ok=True)
    f_csv, f_txt = os.path.join(OUT, "time_shift.csv"), os.path.join(OUT, "time_shift_summary.txt")
    for f in (f_csv, f_txt):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    J = TR.trials()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        d = pd.DataFrame(list(pool.map(one, J)))
    d.to_csv(f_csv, index=False)
    half = d.lag_first_ms - d.lag_second_ms
    per = d.groupby("subject").lag_ms.median()
    out = [f"time_shift.py, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(d)} trials, "
           f"{d.subject.nunique()} subjects, {time.time() - t0:.0f} s. Positive = insole later than the kinematics.",
           f"lag over trials (ms): median {d.lag_ms.median():.1f}, mean {d.lag_ms.mean():.1f}, SD {d.lag_ms.std():.1f}, "
           f"5-95 % {d.lag_ms.quantile(0.05):.1f} .. {d.lag_ms.quantile(0.95):.1f}, min {d.lag_ms.min():.1f}, "
           f"max {d.lag_ms.max():.1f}",
           f"peak inside the lag range in all trials: {bool(d.peak_inside.all())}; peak r median {d.r_peak.median():.3f}, "
           f"min {d.r_peak.min():.3f}",
           f"precision of one estimate: SD(first half - second half) / sqrt 2 = {half.std() / np.sqrt(2):.1f} ms; "
           f"mean half difference {half.mean():.1f} ms",
           "subject medians (ms): " + ", ".join(f"{s} {v:.1f}" for s, v in per.items()),
           f"between-subject SD of subject medians {per.std():.1f} ms; within-subject SD (pooled) "
           f"{np.sqrt(d.groupby('subject').lag_ms.var().mean()):.1f} ms"]
    txt = "\n".join(out)
    print(txt)
    with open(f_txt, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
