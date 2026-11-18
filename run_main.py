"""Main run (Sect. 5, Table 3): the factor graph and the competitors, camera and IMU kinematics.
Common inputs for every method: insole Fz set to 0 below the contact threshold (70 N), insole CoP on the case's own foot
line, whole-body totals Sx, Sz of the case's segment table. The treadmill force plates are read ONLY for scoring
(reference forces; reference joint moments from the camera segment table and the plate forces).
Output: results/main/
Usage: python -I run_main.py test | run [workers] | summary
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

OUT = os.path.join(config.RESULTS, "main")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("main.csv", "main_summary.txt", "main_log.txt"))
JOINTS = ("ankle", "knee", "hip")
METHODS = ("graph", "zero", "newton", "ren", "pivot")
PLATE_CONTACT = 20.0


def rmse(a, b, m):
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if m.sum() > 0 else np.nan


def estimates(kin, ins, BW, dev, pelvis, height, fs):
    """{method: (FxR, FzR, copxR)} with the common inputs; extras of the graph."""
    import competitors as C
    import pivot_split as CP
    thr = dev["F_CONTACT"]
    fz = {s: np.where(ins[s]["Fy"] > thr, ins[s]["Fy"], 0.0) for s in L.SIDES}
    cR = ins["R"]["copx"]
    g = L.Graph(kin, ins, BW, dev)
    out = g.solve()
    est = dict(graph=(out["R"]["Fx"], out["R"]["Fz"], out["R"]["copx"]),
               zero=(np.zeros(len(cR)), fz["R"], cR),
               newton=(C.newton_prop(kin["Sx"], fz["R"], fz["L"])[0], fz["R"], cR),
               ren=(C.sta_split(kin["Sx"], fz["R"], fz["L"], fs)[0], fz["R"], cR))
    f, _ = CP.castellaz(kin["Sx"], kin["Sz"], pelvis, height, {s: dict(Fy=fz[s], copx=ins[s]["copx"]) for s in L.SIDES},
                        fs, f_contact=thr)
    est["pivot"] = (f["R"][0], f["R"][1], cR)
    extra = dict(graph_converged=bool(g.converged), graph_it=g.it, graph_gR=out["gR"], graph_gL=out["gL"],
                 graph_delta_mBW=out["delta"] / BW, n_ds_insole=int(g.ds.sum()), n_contact_insole=int(g.any.sum()))
    return est, extra


def one(args):
    subj, trial = args
    try:
        import planar_model as sp_
        import wang_data as wa
        import wang_segments as wr
        import wang_published as CW
        import camera_mc as K
        R = K.Wang(subj, trial)
        d = wa.load_trial(subj, trial)
        info = wa.subject_info()[subj]
        mass, height, fs = R.mass, R.height, R.FS
        BW = mass * L.G
        _, pelvis_lab, lab_ok = wr.planar_from_markers((None, R.raw), R.n)
        pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, fs)
        rows = []
        for case, dev in (("camera", L.CAMERA), ("imu", L.IMU)):
            kin, ins, _, inf = LI.load(subj, trial, case)
            n = inf["n"]
            pelvis = pelvis_lab[:n] if case == "camera" else inf["P"]["Pelvis"][:n]
            segs = R.segs0 if case == "camera" else None
            if case == "imu":
                import imu_budget as W
                segs = W.imu_inputs(R, subj, trial, inf["tau"])["segs"]
            ref = {s: {k: np.asarray(d["ref"][s][k], float)[:n] for k in ("Fx", "Fy", "copx")} for s in L.SIDES}
            onR, onL = (ref[s]["Fy"] > PLATE_CONTACT for s in L.SIDES)
            win = inf["win"]
            sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
            segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
            ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["R"]["Fx"], ref["R"]["Fy"], ref["R"]["copx"])
            est, extra = estimates(kin, ins, BW, dev, pelvis, height, fs)
            row = dict(subject=subj, trial=trial, case=case, mass=mass, n=n, lab_ok=bool(lab_ok), **extra)
            st, ds, ss = onR & win, onR & onL & win, onR & ~onL & win
            row.update(n_stance=int(st.sum()), n_ds=int(ds.sum()), n_ss=int(ss.sum()))
            for meth, (fx, fzr, cop) in est.items():
                for k, m in (("stance", st), ("ds", ds), ("ss", ss)):
                    row[f"{meth}_{k}"] = 100 * rmse(fx, ref["R"]["Fx"], m) / BW
                row[f"{meth}_fz_stance"] = 100 * rmse(fzr, ref["R"]["Fy"], st) / BW
                mom = sp_.inverse_dynamics(segs, "R", fx, fzr, cop)
                for j in JOINTS:
                    a, b = mom[j][win] / mass, ref_m[j][win] / mass
                    ok = lab_ok and np.isfinite(a).all() and np.isfinite(b).all()
                    row[f"{meth}_{j}"] = float(np.sqrt(np.mean((a - b) ** 2))) if ok else np.nan
                    row[f"{meth}_{j}_r"] = float(np.corrcoef(a, b)[0, 1]) if ok else np.nan
            row["graph_cop_stance_mm"] = 1000 * rmse(est["graph"][2], ref["R"]["copx"], st & (ref["R"]["Fy"] > 0.1 * BW))
            row["insole_cop_stance_mm"] = 1000 * rmse(ins["R"]["copx"], ref["R"]["copx"], st & (ref["R"]["Fy"] > 0.1 * BW))
            if case == "imu":
                T = dict(d=d, n=n, mass=mass, trim=win, ref_m=ref_m, lab_ok=bool(lab_ok))
                zero_row = {f"zero_shear_{k}": np.nan for k in ("stance", "ds", "ss", "vert_stance", "vert_ds", "vert_ss")}
                pub = CW.published(T, zero_row)
                for j in JOINTS:
                    row[f"wang_{j}"], row[f"wang_{j}_r"] = pub[f"Wang_published_{j}"], pub[f"Wang_published_{j}_r"]
            rows.append(row)
        return rows, None
    except Exception:
        return [], f"{subj} {trial}: {traceback.format_exc()}"


def trials():
    import trials as TR
    return TR.trials()


def test():
    rows, err = one(trials()[0])
    if err:
        print(err)
    for r in rows:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


def t95(x):
    """Mean and 95 % interval (t, n - 1) of paired subject differences."""
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return m, m - q * se, m + q * se


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    cols = ("stance", "ds", "ss") + JOINTS
    L_ = [f"run_main.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} subjects, "
          f"{d.groupby(['subject', 'trial']).ngroups} trials. Subject means of trial values. Shear RMSE %BW (right foot, plate "
          f"contact); joint moment RMSE N m/kg over the scoring window.",
          f"graph converged {int(d.graph_converged.sum())}/{len(d)}, iterations median {d.graph_it.median():.0f} (max "
          f"{d.graph_it.max():.0f}); insole double support share median "
          f"{(d.n_ds_insole / d.n).median():.3f}"]
    for case in ("camera", "imu"):
        c = d[d.case == case]
        S = c.groupby("subject").mean(numeric_only=True)
        meths = METHODS + (("wang",) if case == "imu" else ())
        L_ += ["", f"== {case}", "method    " + "  ".join(f"{k:>7s}" for k in cols) + "   " + "  ".join(f"{j + '_r':>7s}" for j in JOINTS)]
        for m in meths:
            vals = [S[f"{m}_{k}"].mean() if f"{m}_{k}" in S else np.nan for k in cols]
            rs = [S[f"{m}_{j}_r"].mean() if f"{m}_{j}_r" in S else np.nan for j in JOINTS]
            L_.append(f"{m:8s}  " + "  ".join(f"{v:7.3f}" for v in vals) + "   " + "  ".join(f"{v:7.3f}" for v in rs))
        L_.append("graph minus the best competitor (lowest subject mean in that column), paired over subjects:")
        for k in cols:
            comp = [m for m in meths if m != "graph" and f"{m}_{k}" in S and np.isfinite(S[f"{m}_{k}"]).any()]
            best = min(comp, key=lambda m: S[f"{m}_{k}"].mean())
            diff = S[f"graph_{k}"] - S[f"{best}_{k}"]
            m_, lo, hi = t95(diff)
            L_.append(f"  {k:7s} best {best:7s} {S[f'{best}_{k}'].mean():.3f}; graph {S[f'graph_{k}'].mean():.3f}; "
                      f"diff {m_:+.3f} (95 % {lo:+.3f}..{hi:+.3f}), {100 * m_ / S[f'{best}_{k}'].mean():+.0f} %, graph better "
                      f"in {int((diff < 0).sum())}/{int(np.isfinite(diff).sum())} subjects")
        cop = (f"graph CoP {S.graph_cop_stance_mm.mean():.1f} mm (insole {S.insole_cop_stance_mm.mean():.1f})" if case == "camera"
               else "CoP not compared (the Xsens horizontal origin is arbitrary)")
        L_.append(f"info: graph Fz RMSE stance {S.graph_fz_stance.mean():.2f} %BW (insole {S.zero_fz_stance.mean():.2f}); "
                  f"{cop}; scale gR "
                  f"{c.graph_gR.median():.3f}, gL {c.graph_gL.median():.3f}; offset {c.graph_delta_mBW.median():+.4f} m BW")
    txt = "\n".join(L_)
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
        res = list(pool.map(one, trials()))
    rows = [r for rr, _ in res for r in rr]
    errs = [e for _, e in res if e]
    pd.DataFrame(rows).to_csv(F_CSV, index=False)
    with open(F_LOG, "x", encoding="utf-8") as fh:
        fh.write(f"run_main.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
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
