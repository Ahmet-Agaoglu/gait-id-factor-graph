"""Ablation of the factor graph (Sect. 5.5, Table 4), camera and IMU kinematics. Each arm removes one part; the force
plates are read for scoring only, as in run_main.py.
Output: results/ablation/
Usage: python -I run_ablation.py run [workers] | summary
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

import factor_graph as L  # noqa: E402
import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "ablation")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("ablation.csv", "ablation_summary.txt", "ablation_log.txt"))
COLS = ("stance", "ds", "ss") + RW.JOINTS
ARMS = (("main", L.FACTORS, {}),
        ("no direction", tuple(f for f in L.FACTORS if f != "direction"), {}),
        ("no force smoothness", tuple(f for f in L.FACTORS if f != "smoothness"), {}),
        ("no moment balance", tuple(f for f in L.FACTORS if f != "moment"), {}),
        ("no scale and offset", L.FACTORS, dict(G_STD=1e-9, D_STD=1e-9)))


def one(args):
    subj, trial = args
    try:
        import planar_model as sp_
        import wang_data as wa
        import wang_segments as wr
        import camera_mc as K
        import imu_budget as W
        R = K.Wang(subj, trial)
        d = wa.load_trial(subj, trial)
        mass, height, fs = R.mass, R.height, R.FS
        BW = mass * L.G
        _, _, lab_ok = wr.planar_from_markers((None, R.raw), R.n)
        rows = []
        for case, dev0 in (("camera", L.CAMERA), ("imu", L.IMU)):
            kin, ins, _, inf = LI.load(subj, trial, case)
            n, win = inf["n"], inf["win"]
            segs = R.segs0 if case == "camera" else W.imu_inputs(R, subj, trial, inf["tau"])["segs"]
            ref = {s: {k: np.asarray(d["ref"][s][k], float)[:n] for k in ("Fx", "Fy", "copx")} for s in L.SIDES}
            onR, onL = (ref[s]["Fy"] > RW.PLATE_CONTACT for s in L.SIDES)
            sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
            segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
            ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["R"]["Fx"], ref["R"]["Fy"], ref["R"]["copx"])
            masks = (("stance", onR & win), ("ds", onR & onL & win), ("ss", onR & ~onL & win))
            for arm, factors, dch in ARMS:
                g = L.Graph(kin, ins, BW, dict(dev0, **dch), factors=factors)
                o = g.solve()
                fx, fz, cop = o["R"]["Fx"], o["R"]["Fz"], o["R"]["copx"]
                row = dict(subject=subj, trial=trial, case=case, arm=arm, converged=bool(g.converged), it=g.it,
                           gR=o["gR"], delta_mBW=o["delta"] / BW)
                row.update({k: 100 * RW.rmse(fx, ref["R"]["Fx"], m) / BW for k, m in masks})
                row["fz_stance"] = 100 * RW.rmse(fz, ref["R"]["Fy"], onR & win) / BW
                mom = sp_.inverse_dynamics(segs, "R", fx, fz, cop)
                for j in RW.JOINTS:
                    a, b = mom[j][win] / mass, ref_m[j][win] / mass
                    ok = lab_ok and np.isfinite(a).all() and np.isfinite(b).all()
                    row[j] = float(np.sqrt(np.mean((a - b) ** 2))) if ok else np.nan
                rows.append(row)
        return rows, None
    except Exception:
        return [], f"{subj} {trial}: {traceback.format_exc()}"


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    main = pd.read_csv(RW.F_CSV)
    lines = [f"run_ablation.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} "
             f"subjects, {d.groupby(['subject', 'trial']).ngroups} trials; converged {int(d.converged.sum())}/{len(d)}. "
             "Subject means; change vs main (plus = error grows when the part is removed, i.e. the part helps); in brackets "
             "the number of subjects where removing the part makes the error larger."]
    for case in ("camera", "imu"):
        chk = d[(d.case == case) & (d.arm == "main")].merge(main[main.case == case], on=["subject", "trial"])
        lines.append(f"check {case}: main vs main.csv max |diff| "
                     f"{max(np.nanmax(np.abs(chk[k] - chk[f'graph_{k}'])) for k in COLS):.2e}")
    lines.append("")
    lines.append(f"{'arm':22s} | camera: shear          moments        Fz      | imu: shear          moments        Fz")
    for arm, _, _ in ARMS[1:]:
        parts = []
        for case in ("camera", "imu"):
            c = d[d.case == case]
            S0 = c[c.arm == "main"].groupby("subject").mean(numeric_only=True)
            S = c[c.arm == arm].groupby("subject").mean(numeric_only=True)
            sh = 100 * (S.stance.mean() / S0.stance.mean() - 1)
            mo = 100 * np.mean([S[j].mean() / S0[j].mean() - 1 for j in RW.JOINTS])
            msub = np.mean([S[j] / S0[j] for j in RW.JOINTS], axis=0)
            fz = 100 * (S.fz_stance.mean() / S0.fz_stance.mean() - 1)
            n_sh = int((S.stance > S0.stance).sum())
            n_mo = int((msub > 1).sum())
            parts.append(f"{sh:+6.1f} % ({n_sh}/9)  {mo:+6.1f} % ({n_mo}/9)  {fz:+6.1f} %")
        lines.append(f"{arm:22s} | " + " | ".join(parts))
    lines.append("")
    lines.append("Detail, subject means per column:")
    for case in ("camera", "imu"):
        g = d[d.case == case].groupby(["arm", "subject"])[list(COLS) + ["fz_stance"]].mean().groupby("arm").mean()
        lines += [f"== {case}", g.round(3).to_string()]
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
        fh.write(f"run_ablation.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
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
