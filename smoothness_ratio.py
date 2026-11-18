"""Horizontal-to-vertical smoothness ratio from the data, plate-free (Sect. 4.1, Table 1). In single support the stance
foot carries the whole ground force, so the kinematic whole-body forces Sx, Sz are that foot's forces. Ratio = RMS second
difference of Sx / that of Sz over single-support frames (one insole above the contact threshold on three consecutive
frames) in the scoring window; both come through the same 6 Hz filter. The value used is the median over trials. For
information only: insole Fz second-difference RMS raw vs 6 Hz filtered, and kinematic Sx second-difference RMS over the
insole vertical scale (raw). No force plate channel is read.
Output: results/smoothness_ratio/
Usage: python -I smoothness_ratio.py run [workers]
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

THR = F.INSOLE["F_CONTACT"]
OUT = os.path.join(config.RESULTS, "smoothness_ratio")


def d2(a, t):
    return a[t] - 2 * a[t + 1] + a[t + 2]


def rms(a):
    return float(np.sqrt(np.mean(np.asarray(a) ** 2)))


def one(args):
    subj, trial, THR = args
    R = K.Wang(subj, trial)
    T = F.whole_body_terms(R.segs0)
    on = {s: R.ins[s]["Fy"] > THR for s in ("R", "L")}
    ss = (on["R"] ^ on["L"]) & R.win
    t = np.where(ss[:-2] & ss[1:-1] & ss[2:])[0]
    fz_raw = np.where(on["R"], R.ins["R"]["Fy"], R.ins["L"]["Fy"])
    fz_f = np.where(on["R"], sp_.lowpass(R.ins["R"]["Fy"], 6.0, R.FS), sp_.lowpass(R.ins["L"]["Fy"], 6.0, R.FS))
    same = on["R"][t] == on["R"][t + 2]
    t = t[same]
    sx, sz = rms(d2(T["Sx"], t)), rms(d2(T["Sz"], t))
    ir, iff = rms(d2(fz_raw, t)), rms(d2(fz_f, t))
    return dict(subject=subj, trial=trial, n_ss=len(t), d2_sx=sx, d2_sz=sz, ratio=sx / sz,
                d2_ins_raw=ir, d2_ins_6hz=iff, ins_filter_ratio=iff / ir, sx_over_ins_raw=sx / ir)


def run(workers=10):
    import trials as TR
    os.makedirs(OUT, exist_ok=True)
    thr = THR
    f_csv, f_txt = (os.path.join(OUT, f"smoothness_ratio{e}") for e in (".csv", "_summary.txt"))
    for f in (f_csv, f_txt):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    J = TR.trials()
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        d = pd.DataFrame(list(pool.map(one, [(s_, t_, thr) for s_, t_ in J])))
    d.to_csv(f_csv, index=False)
    q = lambda c: f"median {d[c].median():.3f} (IQR {d[c].quantile(.25):.3f}-{d[c].quantile(.75):.3f}, " \
                  f"range {d[c].min():.3f}-{d[c].max():.3f})"
    subj = d.groupby("subject")["ratio"].median()
    txt = "\n".join([
        f"smoothness_ratio.py, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} subjects, "
        f"{len(d)} trials, contact threshold {thr:.0f} N, {time.time() - t0:.0f} s; single-support frames per trial median {d.n_ss.median():.0f}.",
        "USED VALUE - ratio d2(Sx)/d2(Sz), kinematic, 6 Hz both: " + q("ratio"),
        f"  subject medians {subj.min():.3f}-{subj.max():.3f}",
        "info - insole Fz d2 RMS, 6 Hz / raw: " + q("ins_filter_ratio"),
        "info - kinematic Sx d2 RMS / insole raw d2 RMS: " + q("sx_over_ins_raw"),
    ])
    print(txt)
    with open(f_txt, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
