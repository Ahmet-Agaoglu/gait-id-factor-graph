"""Where the direction factor acts (Sect. 5.5, Online Resource 1, Table S5). For every trial and both kinematic cases: the
factor graph with and without the direction factor (all values unchanged), scored in more windows than the main run
(vertical force and joint moments in double support, single support and swing; CoP over stance), plus the plate-free
mismatch the direction factor implies. Summed over both feet, with the moment balance, the direction residuals equal
(Hdot - b - d_x S_z) / z_G in double support, so the factor asks the whole-body moment about the CoM to be d_x S_z;
gap = mean over the insole's double-support frames of (Hdot - d_x S_z) / W, from the kinematics alone. The force plates
are read for scoring only, as in run_main.py.
Output: results/direction_factor/
Usage: python -I direction_factor.py run [workers] | summary
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

OUT = os.path.join(config.RESULTS, "direction_factor")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("direction_factor.csv", "direction_factor_summary.txt",
                                                       "direction_factor_log.txt"))
ARMS = (("main", L.FACTORS), ("no direction", tuple(f for f in L.FACTORS if f != "direction")))


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
        for case, dev in (("camera", L.CAMERA), ("imu", L.IMU)):
            kin, ins, _, inf = LI.load(subj, trial, case)
            n, win = inf["n"], inf["win"]
            segs = R.segs0 if case == "camera" else W.imu_inputs(R, subj, trial, inf["tau"])["segs"]
            ref = {s: {k: np.asarray(d["ref"][s][k], float)[:n] for k in ("Fx", "Fy", "copx")} for s in L.SIDES}
            onR, onL = (ref[s]["Fy"] > RW.PLATE_CONTACT for s in L.SIDES)
            sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
            segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
            ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["R"]["Fx"], ref["R"]["Fy"], ref["R"]["copx"])
            st, ds, ss, sw = onR & win, onR & onL & win, onR & ~onL & win, ~onR & win
            # plate-free: the insole's double support and the kinematics only
            g0 = L.Graph(kin, ins, BW, dev)
            D = g0.ds & win
            hd = np.asarray(kin["Hd"], float)[D] / BW
            dsz = L.MODEL["PIVOT_FWD"] * np.asarray(kin["Sz"], float)[D] / BW
            base = dict(subject=subj, trial=trial, case=case, n_ds_insole=int(D.sum()),
                        hdot_ds_mBW=float(hd.mean()), dxsz_ds_mBW=float(dsz.mean()), gap_ds_mBW=float((hd - dsz).mean()))
            for arm, fac in ARMS:
                g = L.Graph(kin, ins, BW, dev, factors=fac)
                o = g.solve()
                fx, fz, cop = o["R"]["Fx"], o["R"]["Fz"], o["R"]["copx"]
                row = dict(base, arm=arm, converged=bool(g.converged), b_mBW=o["delta"] / BW, gR=o["gR"], gL=o["gL"])
                for k, m in (("stance", st), ("ds", ds), ("ss", ss)):
                    row[f"shear_{k}"] = 100 * RW.rmse(fx, ref["R"]["Fx"], m) / BW
                    row[f"fz_{k}"] = 100 * RW.rmse(fz, ref["R"]["Fy"], m) / BW
                row["cop_stance_mm"] = 1000 * RW.rmse(cop, ref["R"]["copx"], st & (ref["R"]["Fy"] > 0.1 * BW))
                mom = sp_.inverse_dynamics(segs, "R", fx, fz, cop)
                for j in RW.JOINTS:
                    a, b = mom[j] / mass, ref_m[j] / mass
                    for k, m in (("all", win), ("ds", ds), ("ss", ss), ("swing", sw)):
                        ok = lab_ok and np.isfinite(a[m]).all() and np.isfinite(b[m]).all()
                        row[f"{j}_{k}"] = float(np.sqrt(np.mean((a[m] - b[m]) ** 2))) if ok and m.sum() else np.nan
                rows.append(row)
        return rows, None
    except Exception:
        return [], f"{subj} {trial}: {traceback.format_exc()}"


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    main = pd.read_csv(RW.F_CSV)
    abl = pd.read_csv(os.path.join(config.RESULTS, "ablation", "ablation.csv"))
    lines = [f"direction_factor.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} "
             f"subjects, {d.groupby(['subject', 'trial']).ngroups} trials; converged {int(d.converged.sum())}/{len(d)}.",
             "Subject means, then the mean over subjects; b and the gap also as medians over trials (m BW)."]
    for case in ("camera", "imu"):
        c = d[d.case == case]
        m0 = c[c.arm == "main"].merge(main[main.case == case], on=["subject", "trial"])
        a0 = c[c.arm == "no direction"].merge(abl[(abl.case == case) & (abl.arm == "no direction")], on=["subject", "trial"])
        lines.append(f"\n== {case}: check vs the main run, max |diff| hip {np.nanmax(np.abs(m0.hip_all - m0.graph_hip)):.1e}, "
                     f"stance shear {np.nanmax(np.abs(m0.shear_stance - m0.graph_stance)):.1e}; vs ablation (no direction) "
                     f"hip {np.nanmax(np.abs(a0.hip_all - a0.hip)):.1e}, b {np.nanmax(np.abs(a0.b_mBW - a0.delta_mBW)):.1e}")
        t = c[c.arm == "main"]
        lines.append(f"gap (Hdot - d_x Sz)/W in insole DS: median over trials {t.gap_ds_mBW.median():+.4f} m BW, subject-mean "
                     f"mean {t.groupby('subject').gap_ds_mBW.mean().mean():+.4f}; Hdot {t.hdot_ds_mBW.median():+.4f}, d_x Sz "
                     f"{t.dxsz_ds_mBW.median():+.4f} (medians); trials with gap < 0: {int((t.gap_ds_mBW < 0).sum())}/{len(t)}")
        for arm in ("main", "no direction"):
            a = c[c.arm == arm]
            lines.append(f"b, {arm}: median {a.b_mBW.median():+.4f} m BW")
        cols = [k for k in d.columns if k.startswith(("shear_", "fz_", "cop_")) or k.split("_")[0] in RW.JOINTS]
        S = c.groupby(["arm", "subject"])[cols].mean().groupby("arm").mean().T
        S["change_%"] = 100 * (S["no direction"] / S["main"] - 1)
        lines.append(S.round(4).to_string())
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
        fh.write(f"direction_factor.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
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
