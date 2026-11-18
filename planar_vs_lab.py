"""Planar reference solution against the data set's own laboratory inverse dynamics (Sect. 3). For every trial: the
reference joint moments (treadmill forces at their CoP through the planar camera model, as in run_main.py) against the
released OpenSim inverse dynamics from the same markers and treadmill (ID.sto), and the released wearable output WP
(ID_Portable.sto) against both. Windows and signs as in run_main.py and wang_published.published. No estimate is involved.
Output: results/planar_vs_lab/
Usage: python -I planar_vs_lab.py run [workers] | summary
"""
import datetime
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "planar_vs_lab")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("planar_vs_lab.csv", "planar_vs_lab_summary.txt",
                                                       "planar_vs_lab_log.txt"))
COLS = dict(ankle="ankle_angle_r_moment", knee="knee_angle_r_moment", hip="hip_flexion_r_moment")


def one(args):
    subj, trial = args
    try:
        import planar_model as sp_
        import wang_data as wa
        import wang_segments as wr
        import camera_mc as K
        import wang_published as CW
        R = K.Wang(subj, trial)
        d = wa.load_trial(subj, trial)
        mass, height, fs = R.mass, R.height, R.FS
        _, _, lab_ok = wr.planar_from_markers((None, R.raw), R.n)
        if not lab_ok:
            return [], None
        _, _, _, inf = LI.load(subj, trial, "camera")
        n, win = inf["n"], inf["win"]
        ref = {k: np.asarray(d["ref"]["R"][k], float)[:n] for k in ("Fx", "Fy", "copx")}
        sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
        segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
        ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["Fx"], ref["Fy"], ref["copx"])
        m = min(n, len(d["id_lab"]), len(d["id_wear"]))
        w = win[:m]
        row = dict(subject=subj, trial=trial, n=int(w.sum()))
        for j in RW.JOINTS:
            pl = ref_m[j][:m][w] / mass
            lab = CW.REF_SIGN[j] * d["id_lab"][COLS[j]].values[:m][w] / mass
            wp = CW.REF_SIGN[j] * d["id_wear"][COLS[j]].values[:m][w] / mass
            for name, (x, y) in (("planar_lab", (pl, lab)), ("wp_planar", (wp, pl)), ("wp_lab", (wp, lab))):
                ok = np.isfinite(x) & np.isfinite(y)
                row[f"{j}_{name}"] = float(np.sqrt(np.mean((x[ok] - y[ok]) ** 2)))
                row[f"{j}_{name}_r"] = float(np.corrcoef(x[ok], y[ok])[0, 1])
        return [row], None
    except Exception:
        return [], f"{subj} {trial}: {traceback.format_exc()}"


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    main = pd.read_csv(RW.F_CSV)
    S = d.groupby("subject").mean(numeric_only=True).mean()
    mi = main[main.case == "imu"].groupby("subject").mean(numeric_only=True).mean()
    lines = [f"planar_vs_lab.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} "
             f"subjects, {len(d)} trials. RMSE in N m/kg (r), subject means then the mean over subjects; frames as the "
             "main run (all scored frames, swing included)."]
    for j in RW.JOINTS:
        lines.append(f"{j:6s} planar vs lab {S[f'{j}_planar_lab']:.3f} ({S[f'{j}_planar_lab_r']:.2f}); WP vs planar "
                     f"{S[f'{j}_wp_planar']:.3f} ({S[f'{j}_wp_planar_r']:.2f}); WP vs lab {S[f'{j}_wp_lab']:.3f} "
                     f"({S[f'{j}_wp_lab_r']:.2f}); check WP vs planar = main run wang_{j}: {mi[f'wang_{j}']:.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


def run(workers=6):
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT, F_LOG):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        res = list(pool.map(one, RW.trials()))
    rows = [r for rr, _ in res for r in rr]
    errs = [e for _, e in res if e]
    pd.DataFrame(rows).to_csv(F_CSV, index=False)
    with open(F_LOG, "x", encoding="utf-8") as fh:
        fh.write(f"planar_vs_lab.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
                 f"{len(errs)} errors, {time.time() - t0:.0f} s\n" + "\n".join(errs) + "\n")
    print(f"{len(rows)} rows, {len(errs)} errors, {time.time() - t0:.0f} s")
    for e in errs[:3]:
        print(e)
    summary()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif sys.argv[1] == "summary":
        summary()
