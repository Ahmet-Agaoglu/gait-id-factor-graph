"""Sensitivity study (Sect. 5.6, Table 5), camera and IMU kinematics. Each value alone at two ends; the inputs of a trial
are read once and every setting is solved on them. Contact-threshold rows recompute the competitors with the same
threshold (common input). The force plates are read for scoring only, as in run_main.py.
Output: results/sensitivity/
Usage: python -I run_sensitivity.py test | run [workers] | summary
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

OUT = os.path.join(config.RESULTS, "sensitivity")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("sensitivity.csv", "sensitivity_summary.txt", "sensitivity_log.txt"))
COLS = ("stance", "ds", "ss") + RW.JOINTS
KIN = ("SX", "SZ", "SH", "D_STD")


def settings():
    """(row, end, model changes, device changes as {key: factor or value})."""
    m = L.MODEL
    S = [("main", "-", {}, {})]
    S += [("pivot height", "low", dict(PIVOT_UP=-0.072), {}), ("pivot height", "high", dict(PIVOT_UP=0.072), {})]
    S += [("pivot forward", "low", dict(PIVOT_FWD=m["PIVOT_FWD"] / 2), {}), ("pivot forward", "high", dict(PIVOT_FWD=m["PIVOT_FWD"] * 2), {})]
    S += [("direction deviation", "low", dict(SDIR=m["SDIR"] / 2), {}), ("direction deviation", "high", dict(SDIR=m["SDIR"] * 2), {})]
    S += [("smoothness ratio", "low", dict(FX_RATIO=m["FX_RATIO"] / 2), {}), ("smoothness ratio", "high", dict(FX_RATIO=m["FX_RATIO"] * 2), {})]
    S += [("kinematic uncertainties", e, {}, {k: f for k in KIN}) for e, f in (("low", 0.5), ("high", 2.0))]
    S += [("insole Fz and scale", e, {}, dict(SI=f, G_STD=f)) for e, f in (("low", 0.5), ("high", 2.0))]
    S += [("CoP", e, {}, dict(SCOP=f)) for e, f in (("low", 0.5), ("high", 2.0))]
    S += [("contact threshold", "low", {}, dict(F_CONTACT=("set", 35.0))), ("contact threshold", "high", {}, dict(F_CONTACT=("set", 140.0)))]
    return S


def apply(dev, ch):
    d = dict(dev)
    for k, f in ch.items():
        d[k] = f[1] if isinstance(f, tuple) else d[k] * f
    return d


def competitors(kin, ins, thr, pelvis, height, fs):
    import competitors as C
    import pivot_split as CP
    fz = {s: np.where(ins[s]["Fy"] > thr, ins[s]["Fy"], 0.0) for s in L.SIDES}
    cR = ins["R"]["copx"]
    est = dict(zero=(np.zeros(len(cR)), fz["R"], cR),
               newton=(C.newton_prop(kin["Sx"], fz["R"], fz["L"])[0], fz["R"], cR),
               ren=(C.sta_split(kin["Sx"], fz["R"], fz["L"], fs)[0], fz["R"], cR))
    f, _ = CP.castellaz(kin["Sx"], kin["Sz"], pelvis, height, {s: dict(Fy=fz[s], copx=ins[s]["copx"]) for s in L.SIDES},
                        fs, f_contact=thr)
    est["pivot"] = (f["R"][0], f["R"][1], cR)
    return est


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
        _, pelvis_lab, lab_ok = wr.planar_from_markers((None, R.raw), R.n)
        pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, fs)
        rows = []
        for case, dev0 in (("camera", L.CAMERA), ("imu", L.IMU)):
            kin, ins, _, inf = LI.load(subj, trial, case)
            n, win = inf["n"], inf["win"]
            pelvis = pelvis_lab[:n] if case == "camera" else inf["P"]["Pelvis"][:n]
            segs = R.segs0 if case == "camera" else W.imu_inputs(R, subj, trial, inf["tau"])["segs"]
            ref = {s: {k: np.asarray(d["ref"][s][k], float)[:n] for k in ("Fx", "Fy", "copx")} for s in L.SIDES}
            onR, onL = (ref[s]["Fy"] > RW.PLATE_CONTACT for s in L.SIDES)
            sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
            segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
            ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["R"]["Fx"], ref["R"]["Fy"], ref["R"]["copx"])
            masks = (("stance", onR & win), ("ds", onR & onL & win), ("ss", onR & ~onL & win))

            def score(fx, fzr, cop):
                out = {k: 100 * RW.rmse(fx, ref["R"]["Fx"], m) / BW for k, m in masks}
                mom = sp_.inverse_dynamics(segs, "R", fx, fzr, cop)
                for j in RW.JOINTS:
                    a, b = mom[j][win] / mass, ref_m[j][win] / mass
                    ok = lab_ok and np.isfinite(a).all() and np.isfinite(b).all()
                    out[j] = float(np.sqrt(np.mean((a - b) ** 2))) if ok else np.nan
                return out
            for row, end, mch, dch in settings():
                dev = apply(dev0, dch)
                model = dict(L.MODEL, **mch)
                g = L.Graph(kin, ins, BW, dev, model=model)
                o = g.solve()
                base = dict(subject=subj, trial=trial, case=case, row=row, end=end)
                rows.append(dict(base, method="graph", converged=bool(g.converged), it=g.it,
                                 **score(o["R"]["Fx"], o["R"]["Fz"], o["R"]["copx"])))
                if row == "contact threshold":
                    for meth, e in competitors(kin, ins, dev["F_CONTACT"], pelvis, height, fs).items():
                        rows.append(dict(base, method=meth, **score(*e)))
        return rows, None
    except Exception:
        return [], f"{subj} {trial}: {traceback.format_exc()}"


def test():
    rows, err = one(RW.trials()[0])
    if err:
        print(err)
    d = pd.DataFrame(rows)
    main = pd.read_csv(RW.F_CSV)
    t = RW.trials()[0]
    for case in ("camera", "imu"):
        a = d[(d.case == case) & (d.row == "main") & (d.method == "graph")].iloc[0]
        b = main[(main.subject == t[0]) & (main.trial == t[1]) & (main.case == case)].iloc[0]
        print(case, "main setting vs the main run, max |diff|:", max(abs(a[k] - b[f"graph_{k}"]) for k in COLS))
    print(d[d.method == "graph"][["case", "row", "end", "it", "stance", "hip"]].round(3).to_string())


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    main = pd.read_csv(RW.F_CSV)
    lines = [f"run_sensitivity.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} subjects, "
             f"{d.groupby(['subject', 'trial']).ngroups} trials; graph converged {int(d[d.method == 'graph'].converged.sum())}/"
             f"{int((d.method == 'graph').sum())}. Subject means; change vs the main setting.",
             "Columns per case: stance shear change %, mean change % of the three joint-moment errors, and the columns where the "
             "graph led in the main run but no longer beats the best competitor ('-' = still ahead in all)."]
    lead = dict(camera=COLS, imu=("stance", "ds", "ss", "knee", "hip"))
    comp_main = {}
    for case in ("camera", "imu"):
        S = main[main.case == case].groupby("subject").mean(numeric_only=True)
        comp_main[case] = {k: min(S[f"{m}_{k}"].mean() for m in ("zero", "newton", "ren", "pivot", "wang") if f"{m}_{k}" in S
                                  and np.isfinite(S[f"{m}_{k}"]).any()) for k in COLS}
        g0 = d[(d.case == case) & (d.row == "main") & (d.method == "graph")]
        chk = g0.merge(main[main.case == case], on=["subject", "trial"])
        lines.append(f"check {case}: main setting vs main.csv, max |diff| "
                     f"{max(np.nanmax(np.abs(chk[k] - chk[f'graph_{k}'])) for k in COLS):.2e}")
    lines.append("")
    lines.append(f"{'row':26s} {'end':5s} | camera: shear  moments  lost      | imu: shear  moments  lost")
    for row, end, _, _ in settings()[1:]:
        parts = []
        for case in ("camera", "imu"):
            c = d[(d.case == case)]
            G0 = c[(c.row == "main") & (c.method == "graph")].groupby("subject")[list(COLS)].mean().mean()
            sel = c[(c.row == row) & (c.end == end)]
            G = sel[sel.method == "graph"].groupby("subject")[list(COLS)].mean().mean()
            if row == "contact threshold":
                Cm = sel[sel.method != "graph"].groupby(["method", "subject"])[list(COLS)].mean().groupby("method").mean()
                best = Cm.min()
                if case == "imu":
                    best[list(RW.JOINTS)] = np.minimum(best[list(RW.JOINTS)].values,
                                                       [main[main.case == "imu"].groupby("subject")[f"wang_{j}"].mean().mean()
                                                        for j in RW.JOINTS])
            else:
                best = pd.Series(comp_main[case])
            lost = [k for k in lead[case] if not G[k] < best[k]]
            dsh = 100 * (G["stance"] / G0["stance"] - 1)
            dm = 100 * np.mean([G[j] / G0[j] - 1 for j in RW.JOINTS])
            parts.append(f"{dsh:+6.1f} %  {dm:+6.1f} %  {','.join(lost) if lost else '-':8s}")
        lines.append(f"{row:26s} {end:5s} | " + " | ".join(parts))
    lines.append("")
    lines.append("Detail, subject means per column (graph):")
    for case in ("camera", "imu"):
        g = d[(d.case == case) & (d.method == "graph")].groupby(["row", "end", "subject"])[list(COLS)].mean() \
            .groupby(["row", "end"]).mean()
        lines.append(f"== {case}")
        lines.append(g.round(3).to_string())
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
        fh.write(f"run_sensitivity.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
                 f"{len(errs)} errors, {time.time() - t0:.0f} s\n" + "\n".join(errs) + "\n")
    print(f"{len(rows)} rows, {len(errs)} errors, {time.time() - t0:.0f} s")
    for e in errs[:3]:
        print(e)
    summary()


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "test":
        test()
    elif sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif sys.argv[1] == "summary":
        summary()
