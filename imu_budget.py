"""IMU kinematics: the device-dependent uncertainties from the difference between the IMU and camera kinematics
(Sect. 4.2, Table 2), and the IMU inputs of the graph (imu_inputs).
IMU reading (the method's own path, plate-free): Xsens MVNX joint points, time-aligned to the summed insole force
(align_insole), 6 Hz filter, the same planar segment model as the camera (wang_segments.build), kinematic terms by
factor_graph.whole_body_terms; insole CoP on the Xsens heel -> toe line.
For the uncertainty only, the camera case is the comparison: the IMU frame is shifted to the camera frame by a constant and
linear offset that matches the CoM horizontal position (the graph is translation invariant). Difference e = IMU - camera,
statistics as camera_mc (t, c, v over the scored window). Shared errors (segment parameters, rigid trunk model) cancel in
e; non-shared camera errors (noise, soft tissue, landmarks) add. Per trial: IMU^2 = e^2 - camera_nonshared^2 + shared^2,
camera values from results/camera_mc; CoP likewise (camera non-shared: markers, time; shared: insole displacement in the
shoe; results/cop_mc). Diagnostic only: lag between IMU and camera Sz after the alignment. No force plate channel is read.
Output: results/imu_budget/
Usage: python -I imu_budget.py test | run [workers]
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
import time_shift as TS  # noqa: E402

THR = F.INSOLE["F_CONTACT"]     # contact threshold of the graph
SENSOR_COP = 0.015
OUT = os.path.join(config.RESULTS, "imu_budget")
XS = {"Hip": "UpperLeg", "Knee": "LowerLeg", "Ankle": "Foot", "MT5": "Toe"}


R_MIN = 0.5      # same reliability threshold as the camera time shift (time_shift.py)


def insole_fz(R):
    return tuple(np.where(R.ins[s]["Fy"] > THR, R.ins[s]["Fy"], 0.0) for s in ("R", "L"))


def align_insole(subj, trial, fzR, fzL, n, fs, span=4.0, coarse=0.02, fine=0.004):
    """MVNX time offset from the insoles only. Timing: the total-force match of wang_data.align_time,
    corr(a_com,z, total insole Fz). It repeats every step and can lock one step off with the feet swapped, so only offsets
    where the side term corr(left - right ankle height (IMU), right - left insole Fz) is positive are allowed. The side
    term is not used for timing (its peak lies 0.1-0.25 s early). Returns tau, corr_total, corr_side."""
    import wang_data as wa
    mv = wa.parse_mvnx(wa.trial_paths(subj, trial)["mvnx"])
    ts, acc = mv["t"], mv["com_acc"][:, 1]
    dh = mv["pos"]["LeftFoot"][:, 1] - mv["pos"]["RightFoot"][:, 1]
    sl = slice(int(fs), n - int(fs))
    tot, side = (fzR + fzL)[sl], (fzR - fzL)[sl]

    def score(tau):
        td = np.arange(n) / fs + tau
        a, h = np.interp(td, ts, acc)[sl], np.interp(td, ts, dh)[sl]
        if min(np.std(a), np.std(h), np.std(tot), np.std(side)) < 1e-9:
            return -1.0, -1.0
        return np.corrcoef(a, tot)[0, 1], np.corrcoef(h, side)[0, 1]

    def key(x):
        c1, c2 = score(x)
        return c1 if c2 > 0 else -2.0
    best = max(np.arange(-span, span + 1e-9, coarse), key=key)
    best = max(np.arange(best - coarse, best + coarse + 1e-9, fine), key=key)
    return (float(best),) + tuple(float(c) for c in score(best))


def align_one(args):
    subj, trial = args
    R = K.Wang(subj, trial)
    tau, c1, c2 = align_insole(subj, trial, *insole_fz(R), R.n, R.FS)
    return dict(subject=subj, trial=trial, tau_own=tau, r_align=c1, r_side=c2)


def imu_inputs(R, subj, trial, tau):
    """The IMU case's own inputs for trial R (camera_mc.Wang) at the MVNX offset tau: segment table, joint points, insole CoP."""
    import wang_data as wa
    import wang_segments as wr
    k = wa.xsens_planar(subj, trial, R.n, tau=tau)
    m = min(R.n, k["n"])
    P = {f"{s}.{j}": k["pos"][("Right" if s == "R" else "Left") + x] for s in ("R", "L") for j, x in XS.items()}
    P["Pelvis"] = k["pos"]["Pelvis"]
    P = {key: sp_.lowpass(v[:m], 6.0, R.FS) for key, v in P.items()}
    segs = wr.build(P, R.mass, R.height, m)
    cop = {s: wr.insole_cop_to_lab(P, s, R.ins[s]["copx"], m) for s in ("R", "L")}
    return dict(m=m, segs=segs, P=P, cop=cop)


def one(args):
    import wang_segments as wr
    subj, trial, tau = args
    R = K.Wang(subj, trial)
    I = imu_inputs(R, subj, trial, tau)
    m, fs, BW = I["m"], R.FS, R.mass * F.G
    win = R.win[:m].copy()
    win[m - int(fs):] = False
    Kt = {key: v[:m] for key, v in F.whole_body_terms(R.segs0).items()}
    Kp = F.whole_body_terms(I["segs"])
    t = np.arange(m) / fs
    A = np.c_[np.ones(win.sum()), t[win]]
    ab = np.linalg.lstsq(A, (Kt["xc"] - Kp["xc"])[win], rcond=None)[0]
    shift = ab[0] + ab[1] * t
    Kp["xc"] = Kp["xc"] + shift
    e = K.errors(Kp, Kt, BW)
    row = dict(subject=subj, trial=trial, n=m, n_short=R.n - m, tau_s=tau, frame_offset_m=float(ab[0]), frame_drift_mps=float(ab[1]))
    for q in K.QTY:
        row[f"{q}_t"], row[f"{q}_c"], row[f"{q}_v"] = K.stats(e[q][None, :], win)
    tot, con, var = [], [], []
    for s in ("R", "L"):
        on = (R.ins[s]["Fy"][:m] > THR) & win
        cam = wr.insole_cop_to_lab(R.P0, s, R.ins[s]["copx"], m, heel_key=s + ".Heel")
        d = (I["cop"][s] + shift - cam)[on]
        c = d.mean()
        tot.append(np.mean(d ** 2))
        con.append(c ** 2)
        var.append(np.mean((d - c) ** 2))
    row["cop_t"], row["cop_c"], row["cop_v"] = (float(np.sqrt(np.mean(a))) for a in (tot, con, var))
    idx = np.where(win)[0]
    idx = idx[(idx - TS.MAXLAG >= 0) & (idx + TS.MAXLAG < m)]
    lag, r, inside = TS.lag_of(Kt["Sz"], Kp["Sz"], idx)
    row["lag_imu_cam_ms"], row["lag_r"], row["lag_inside"] = lag * 1000.0 / fs, r, inside
    return row


def combine(d):
    """IMU^2 = diff^2 - camera non-shared^2 + shared^2, per trial."""
    cam = pd.read_csv(K.F_CSV)
    cop = pd.read_csv(os.path.join(config.RESULTS, "cop_mc", "cop_mc.csv"))
    g = d.merge(cam, on=["subject", "trial"], suffixes=("", "_cam")).merge(cop, on=["subject", "trial"])
    out = pd.DataFrame(dict(subject=g.subject, trial=g.trial))
    neg = {}
    for name, q, st in (("SX", "x", "t"), ("SZ", "z", "t"), ("SH", "m", "v"), ("D_STD", "m", "c")):
        ns = sum(g[f"{q}_{src}_{st}"] ** 2 for src in ("noise", "sta", "land"))
        sh = sum(g[f"{q}_{src}_{st}"] ** 2 for src in ("bsip", "meas"))
        v2 = g[f"{q}_{st}"] ** 2 - ns + sh
        neg[name] = int((v2 < 0).sum())
        out[name] = np.sqrt(v2.clip(lower=0))
        out[name + "_diff"] = g[f"{q}_{st}"]
        out[name + "_cam"] = np.sqrt(ns + sh)
    v2 = g.cop_t ** 2 - g.cop_mark_t ** 2 - g.cop_time_t ** 2 + g.cop_shoe_t ** 2
    neg["COP"] = int((v2 < 0).sum())
    out["COP_place"] = np.sqrt(v2.clip(lower=0))
    out["COP_diff"] = g.cop_t
    out["COP_total"] = np.sqrt(SENSOR_COP ** 2 + out["COP_place"] ** 2)
    return out, neg


def trials():
    import trials as TR
    return TR.trials()


def test():
    a = align_one(trials()[0])
    r = one((a["subject"], a["trial"], a["tau_own"]))
    print(a, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def alignments(pool):
    """Own offset where the total-force match is reliable (r >= R_MIN); else the median offset of the reliable trials."""
    a = pd.DataFrame(list(pool.map(align_one, trials())))
    a["reliable"] = a.r_align >= R_MIN
    a["tau"] = np.where(a.reliable, a.tau_own, a.loc[a.reliable, "tau_own"].median())
    return a


def subj_range(v, col):
    s = v.groupby("subject")[col].median()
    return f"{v[col].median():.4f} (subjects {s.min():.4f}-{s.max():.4f})"


def run(workers=6):
    os.makedirs(OUT, exist_ok=True)
    fs = dict(raw=os.path.join(OUT, "imu_budget_trials.csv"), val=os.path.join(OUT, "imu_budget_values.csv"),
              txt=os.path.join(OUT, "imu_budget_summary.txt"))
    t0 = time.time()
    if workers == 0:                      # summary from the saved per-trial file (combine step only)
        d = pd.read_csv(fs["raw"])
        for k in ("val", "txt"):
            if os.path.exists(fs[k]):
                raise SystemExit("exists: " + fs[k])
    else:
        for f in fs.values():
            if os.path.exists(f):
                raise SystemExit("exists: " + f)
        with ProcessPoolExecutor(max_workers=workers) as pool:
            a = alignments(pool)
            d = pd.DataFrame(list(pool.map(one, [tuple(x) for x in a[["subject", "trial", "tau"]].values])))
        d = a.drop(columns="tau").merge(d, on=["subject", "trial"])
        d.to_csv(fs["raw"], index=False)
    v, neg = combine(d)
    v.to_csv(fs["val"], index=False)
    rel = d[d.reliable]
    L = ["wang_imu_budget.py, " + datetime.datetime.now().isoformat(timespec="seconds") +
         f": {d.subject.nunique()} subjects, {len(d)} trials, {time.time() - t0:.0f} s. Medians over trials.",
         f"alignment to the insole: reliable (total r >= {R_MIN}) {len(rel)}/{len(d)}; their offset median "
         f"{rel.tau_own.median():.3f} s (range {rel.tau_own.min():.3f}-{rel.tau_own.max():.3f}), used for the others "
         f"({', '.join(d.loc[~d.reliable, 'subject'] + ' ' + d.loc[~d.reliable, 'trial'])}); total r median "
         f"{d.r_align.median():.3f}, side r min {d.r_side.min():.3f}; frames short max {int(d.n_short.max())}",
         f"diagnostic, IMU vs camera Sz lag after alignment: median {d.lag_imu_cam_ms.median():.1f} ms, RMS "
         f"{np.sqrt(np.mean(d.lag_imu_cam_ms ** 2)):.1f} ms, range {d.lag_imu_cam_ms.min():.1f}..{d.lag_imu_cam_ms.max():.1f}, "
         f"peak inside range {int(d.lag_inside.sum())}/{len(d)}, r median {d.lag_r.median():.3f}",
         "", "value   IMU                               difference  camera(total)  negative"]
    for name in ("SX", "SZ", "SH", "D_STD"):
        L.append(f"{name:6s}  {subj_range(v, name)}  {v[name + '_diff'].median():.4f}      "
                 f"{v[name + '_cam'].median():.4f}         {neg[name]}")
    L.append(f"CoP placement (m): IMU {subj_range(v, 'COP_place')}, difference {v.COP_diff.median():.4f}, "
             f"negative {neg['COP']}; total with the sensor 15 mm: {v.COP_total.median():.4f}")
    txt = "\n".join(L)
    print(txt)
    with open(fs["txt"], "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
