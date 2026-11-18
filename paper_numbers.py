"""Numbers of the article, recomputed from the result files. One source for the tables (paper_tables.py) and for the check of
the text (verify.py). Reads results only, writes nothing.
Every value is a subject mean of trial values unless stated (as the scoring scripts): mean over trials per subject, then mean
over the nine subjects.
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

R = config.RESULTS
JOINTS = ("ankle", "knee", "hip")
SHEAR = ("stance", "ds", "ss")
COLS = SHEAR + JOINTS
CASES = ("camera", "imu")
METHODS = ("zero", "newton", "ren", "pivot", "wang", "graph")
ABL = ("no direction", "no force smoothness", "no moment balance", "no scale and offset", "balances and sensors only")
SENS = ("pivot height", "pivot forward", "direction deviation", "smoothness ratio", "kinematic uncertainties",
        "insole Fz and scale", "CoP", "contact threshold", "smoothness")


def path(*p):
    return os.path.join(R, *p)


def t95(x):
    from scipy import stats
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    q = stats.t.ppf(0.975, len(x) - 1)
    return m, m - q * se, m + q * se


def subj(d, case):
    return d[d.case == case].groupby("subject").mean(numeric_only=True)


def main_results():
    """Table of subject means, method x column, per case; graph minus best competitor per column."""
    d = pd.read_csv(path("main", "main.csv"))
    out = dict(n_trials=d.groupby(["subject", "trial"]).ngroups, n_subjects=d.subject.nunique(),
               converged=int(d.graph_converged.sum()), n_solves=len(d), it_median=float(d.graph_it.median()),
               it_max=int(d.graph_it.max()), ds_share=float((d.n_ds_insole / d.n).median()),
               frames_median=float(d[d.case == "camera"].n.median()))
    tab, con = {}, {}
    for case in CASES:
        S = subj(d, case)
        for m in METHODS:
            for k in COLS + ("fz_stance",):
                c = f"{m}_{k}"
                tab[(case, m, k)] = float(S[c].mean()) if c in S and np.isfinite(S[c]).any() else np.nan
        for k in COLS:
            comp = [m for m in METHODS if m != "graph" and np.isfinite(tab[(case, m, k)])]
            best = min(comp, key=lambda m: tab[(case, m, k)])
            diff = S[f"graph_{k}"] - S[f"{best}_{k}"]
            mm, lo, hi = t95(diff)
            con[(case, k)] = dict(best=best, best_val=tab[(case, best, k)], diff=mm, lo=lo, hi=hi,
                                  pct=100 * mm / tab[(case, best, k)], n_better=int((diff < 0).sum()),
                                  n=int(np.isfinite(diff).sum()))
        c = d[d.case == case]
        out[(case, "gR")], out[(case, "gL")] = float(c.graph_gR.median()), float(c.graph_gL.median())
        out[(case, "delta")] = float(c.graph_delta_mBW.median())
        out[(case, "fz_graph")] = tab[(case, "graph", "fz_stance")]
        out[(case, "fz_insole")] = tab[(case, "zero", "fz_stance")]
        out[(case, "cop_graph")] = float(S.graph_cop_stance_mm.mean())
        out[(case, "cop_insole")] = float(S.insole_cop_stance_mm.mean())
        diff = S.graph_cop_stance_mm - S.insole_cop_stance_mm
        mm, lo, hi = t95(diff)
        con[(case, "cop")] = dict(best="insole", best_val=out[(case, "cop_insole")], diff=mm, lo=lo, hi=hi,
                                  pct=100 * mm / out[(case, "cop_insole")], n_better=int((diff < 0).sum()),
                                  n=int(np.isfinite(diff).sum()))
    out["table"], out["contrast"] = tab, con
    return out


def _arm_change(d, case, arm):
    c = d[d.case == case]
    S0 = c[c.arm == "main"].groupby("subject").mean(numeric_only=True)
    S = c[c.arm == arm].groupby("subject").mean(numeric_only=True)
    sh = 100 * (S.stance.mean() / S0.stance.mean() - 1)
    mo = 100 * np.mean([S[j].mean() / S0[j].mean() - 1 for j in JOINTS])
    msub = np.mean([S[j] / S0[j] for j in JOINTS], axis=0)
    fz = 100 * (S.fz_stance.mean() / S0.fz_stance.mean() - 1)
    return dict(shear=sh, moments=mo, n_shear=int((S.stance > S0.stance).sum()), n_moments=int((msub > 1).sum()), fz=fz,
                abs={k: float(S[k].mean()) for k in COLS + ("fz_stance",)})


def ablation():
    d1 = pd.read_csv(path("ablation", "ablation.csv"))
    d2 = pd.read_csv(path("ablation2", "ablation2.csv"))
    out = {}
    for case in CASES:
        for arm in ABL:
            d = d2 if arm == "balances and sensors only" else d1
            out[(case, arm)] = _arm_change(d, case, arm)
    return out


def sensitivity():
    """Per row and end: shear and mean moment change vs main, and the columns where the graph led the best competitor in the
    main run but no longer does (as run_sensitivity.summary)."""
    d = pd.read_csv(path("sensitivity", "sensitivity.csv"))
    sm = pd.read_csv(path("sensitivity_smoothness", "sensitivity_smoothness.csv"))           # sigma_sm halved and doubled (run_sensitivity_smoothness.py)
    d = pd.concat([d, sm[sm.row == "smoothness"]], ignore_index=True)
    main = pd.read_csv(path("main", "main.csv"))
    lead = dict(camera=COLS, imu=("stance", "ds", "ss", "knee", "hip"))
    out = {}
    for case in CASES:
        S = main[main.case == case].groupby("subject").mean(numeric_only=True)
        comp_main = pd.Series({k: min(S[f"{m}_{k}"].mean() for m in METHODS if m != "graph" and f"{m}_{k}" in S
                                      and np.isfinite(S[f"{m}_{k}"]).any()) for k in COLS})
        c = d[d.case == case]
        G0 = c[(c.row == "main") & (c.method == "graph")].groupby("subject")[list(COLS)].mean().mean()
        for row in SENS:
            for end in ("low", "high"):
                sel = c[(c.row == row) & (c.end == end)]
                G = sel[sel.method == "graph"].groupby("subject")[list(COLS)].mean().mean()
                if row == "contact threshold":
                    best = sel[sel.method != "graph"].groupby(["method", "subject"])[list(COLS)].mean().groupby("method") \
                        .mean().min()
                    if case == "imu":
                        best[list(JOINTS)] = np.minimum(best[list(JOINTS)].values, [S[f"wang_{j}"].mean() for j in JOINTS])
                else:
                    best = comp_main
                lost = [k for k in lead[case] if not G[k] < best[k]]
                out[(case, row, end)] = dict(shear=100 * (G["stance"] / G0["stance"] - 1),
                                             moments=100 * np.mean([G[j] / G0[j] - 1 for j in JOINTS]), lost=lost,
                                             abs={k: float(G[k]) for k in COLS})
    return out


def derived():
    """Values of the uncertainty table that come from Wang's own data (kinematics and insoles; never the plates)."""
    out = {}
    s = pd.read_csv(path("smoothness_ratio", "smoothness_ratio.csv"))
    out["ratio_median"] = float(s.ratio.median())
    out["ratio_q1"], out["ratio_q3"] = (float(s.ratio.quantile(q)) for q in (0.25, 0.75))
    cop = pd.read_csv(path("cop_mc", "cop_mc.csv"))
    out["cop_place_cam_mm"] = 1000 * float(cop.cop_all_t.median())
    out["cop_total_cam_mm"] = 1000 * float(np.median(np.sqrt(cop.cop_all_t ** 2 + 0.015 ** 2)))
    import camera_mc as K
    cam = K.graph_values(pd.read_csv(path("camera_mc", "camera_mc.csv")))
    for key, name in (("SX", "cam_SX"), ("SZ", "cam_SZ"), ("SH", "cam_SH"), ("D_STD", "cam_D")):
        out[name] = round(float(cam[key].median()), 4)
    v = pd.read_csv(path("imu_budget", "imu_budget_values.csv"))
    for k in ("SX", "SZ", "SH", "D_STD"):
        out["imu_" + k] = float(v[k].median())
    out["imu_cop_place_mm"] = 1000 * float(v.COP_place.median())
    out["imu_cop_total_mm"] = 1000 * float(v.COP_total.median())
    a = pd.read_csv(path("imu_budget", "imu_budget_trials.csv"))
    out["imu_reliable"] = int(a.reliable.sum())
    out["imu_tau_median"] = float(a.loc[a.reliable, "tau_own"].median())
    th = pd.read_csv(path("contact_threshold", "contact_threshold.csv"))
    g = th.groupby("thr")[["rms_ms", "ds_share", "contact_share"]].median()
    out["thr_chosen"] = float(g.rms_ms.idxmin())
    out["thr_rms_ms"] = float(g.rms_ms.min())
    sm = pd.read_csv(path("smoothness_scale", "smoothness_scale.csv"))
    out["sm_median_N"] = float(sm[sm.case == "camera"].sigma_sm_N.median())
    out["sm_median_N_imu"] = float(sm[sm.case == "imu"].sigma_sm_N.median())
    ts = pd.read_csv(path("time_shift", "time_shift.csv"))
    rel = ts[ts.r_peak >= 0.5]
    out["shift_n"] = len(rel)
    out["shift_rms_ms"] = float(np.sqrt(np.mean(rel.lag_ms ** 2)))
    return out


def timing():
    d = pd.read_csv(path("timing", "timing.csv"))
    return dict(seconds_median=float(d.seconds.median()), seconds_max=float(d.seconds.max()),
                unknowns_median=float(d.unknowns.median()), frames_median=float(d.frames.median()))


def speed():
    """Per case, speed and column: FG and the closest competitor of the main run (by_speed.py)."""
    d = pd.read_csv(path("by_speed", "by_speed.csv"))
    return {(r.case, float(r.speed), r.col): dict(comp=r.comp, graph=r.graph, other=r.other, pct=r.pct, n_better=int(r.n_better),
                                                  n=int(r.n)) for r in d.itertuples()}


def direction():
    """With and without the direction factor (direction_factor.py): subject means per arm and column, and b and the gap of
    double support as medians over trials (m BW)."""
    d = pd.read_csv(path("direction_factor", "direction_factor.csv"))
    cols = [c for c in d.columns if c.split("_")[0] in ("shear", "fz", "cop", "ankle", "knee", "hip")]
    out = {}
    for case in CASES:
        c = d[d.case == case]
        for arm in ("main", "no direction"):
            a = c[c.arm == arm]
            S = a.groupby("subject")[cols].mean().mean()
            out[(case, arm)] = {k: float(S[k]) for k in cols}
            out[(case, arm, "b")] = float(a.b_mBW.median())
        m = c[c.arm == "main"]
        out[(case, "gap")] = float(m.gap_ds_mBW.median())
        out[(case, "gap_neg")] = int((m.gap_ds_mBW < 0).sum())
        out[(case, "n")] = len(m)
    return out


def planar_lab():
    """Planar reference vs the released 3-D laboratory moments, and WP against each (planar_vs_lab.py); RMSE N m/kg,
    subject means then the mean over subjects."""
    d = pd.read_csv(path("planar_vs_lab", "planar_vs_lab.csv"))
    S = d.groupby("subject").mean(numeric_only=True).mean()
    return {(j, k): float(S[f"{j}_{k}"]) for j in JOINTS for k in ("planar_lab", "wp_planar", "wp_lab")}


def hip():
    """Effect of not using Harrington's anterior-posterior term (hip_center.py): medians over trials."""
    d = pd.read_csv(path("hip_center", "hip_center.csv"))
    return {k: float(d[k].median()) for k in ("pd_mm", "shift_mm", "shift_x_mm", "dxG_mm", "moment_mBW")}


def all_numbers():
    return dict(main=main_results(), abl=ablation(), sens=sensitivity(), der=derived(), time=timing(), speed=speed(),
                dirn=direction(), plab=planar_lab(), hip=hip())


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    N = all_numbers()
    m = N["main"]
    print({k: v for k, v in m.items() if k not in ("table", "contrast")})
    for (case, k), c in m["contrast"].items():
        print(case, k, {q: (round(v, 3) if isinstance(v, float) else v) for q, v in c.items()})
    for key, v in N["abl"].items():
        print(key, {q: (round(x, 1) if isinstance(x, float) else x) for q, x in v.items() if q != "abs"})
    for key, v in N["sens"].items():
        print(key, round(v["shear"], 1), round(v["moments"], 1), v["lost"])
    print(N["der"])
