"""Inputs of the factor graph (factor_graph.py) for one trial of Wang et al. (2023), camera or IMU kinematics.
Both cases read their joint points through the same 6 Hz filter and the same planar segment model; the insole Fz and
CoP (Moticon) are the same; the insole CoP is placed on each case's own heel -> toe line. IMU (Xsens) time is aligned to the
insoles (imu_budget.align_insole; offsets in results/imu_budget: own offset where the total-force match is
reliable, else the median of the reliable trials). No force plate channel is read.
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import pandas as pd  # noqa: E402

import factor_graph as L  # noqa: E402
import camera_mc as K  # noqa: E402

ALIGN_CSV = os.path.join(config.RESULTS, "imu_budget", "imu_budget_trials.csv")


def imu_offset(subj, trial):
    a = pd.read_csv(ALIGN_CSV)
    tau_med = a.loc[a.reliable, "tau_own"].median()
    row = a[(a.subject == subj) & (a.trial == trial)].iloc[0]
    return float(row.tau_own if row.reliable else tau_med)


def load(subj, trial, case):
    """(kin, ins, BW, info) for case 'camera' or 'imu'."""
    import wang_segments as wr
    R = K.Wang(subj, trial)
    BW = R.mass * L.G
    if case == "camera":
        n = R.n
        kin = L.whole_body_terms(R.segs0)
        P = R.P0
        cop = {s: wr.insole_cop_to_lab(P, s, R.ins[s]["copx"], n, heel_key=s + ".Heel") for s in L.SIDES}
        info = dict(heel={s: P[s + ".Heel"][:n] for s in L.SIDES})
    elif case == "imu":
        import imu_budget as W
        tau = imu_offset(subj, trial)
        I = W.imu_inputs(R, subj, trial, tau)
        n = I["m"]
        kin = L.whole_body_terms(I["segs"])
        P = I["P"]
        cop = I["cop"]
        info = dict(tau=tau)
    else:
        raise ValueError(case)
    ins = {s: dict(Fy=R.ins[s]["Fy"][:n], copx=cop[s][:n]) for s in L.SIDES}
    info.update(n=n, win=R.win[:n].copy(), P=P, mass=R.mass)
    info["win"][n - int(R.FS):] = False
    return kin, ins, BW, info
