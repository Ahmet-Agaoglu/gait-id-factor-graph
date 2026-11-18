"""Data of Fig. 2: one trial, camera and IMU kinematics; right-foot shear and hip moment over three strides; reference
solution, pivot split (PS) and factor graph (FG). The trial is the one closest to typical: the smallest sum, over both
kinematic cases and over the stance shear and the hip moment, of the distance of its FG error's percentile rank from 0.5
(results/main/main.csv). The trial is recomputed exactly as in run_main.py, and its errors are
checked against that file. The force plates enter the reference only.
Output: results/figure2/ (plotted series); figure2.py draws the figure.
Usage: python -I figure2_data.py
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

import factor_graph as L  # noqa: E402
import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

OUT_DIR = os.path.join(config.RESULTS, "figure2")
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
C_FG, C_PS = "#2a78d6", "#eb6834"


def pick_trial():
    """The trial closest to typical: smallest sum over (camera, IMU) x (stance shear, hip) of |percentile rank - 0.5| of
    its FG error."""
    d = pd.read_csv(RW.F_CSV)
    score = None
    for case in ("camera", "imu"):
        c = d[d.case == case].set_index(["subject", "trial"])
        for k in ("graph_stance", "graph_hip"):
            p = (c[k].rank(pct=True) - 0.5).abs()
            score = p if score is None else score + p
    subj, trial = score.dropna().idxmin()
    return subj, trial, d


def series(subj, trial):
    import camera_mc as K
    import planar_model as sp_
    import wang_data as wa
    import wang_segments as wr
    import imu_budget as W
    R = K.Wang(subj, trial)
    d = wa.load_trial(subj, trial)
    mass, height, fs = R.mass, R.height, R.FS
    BW = mass * L.G
    _, pelvis_lab, lab_ok = wr.planar_from_markers((None, R.raw), R.n)
    pelvis_lab = sp_.lowpass(pelvis_lab, 6.0, fs)
    out = {}
    for case, dev in (("camera", L.CAMERA), ("imu", L.IMU)):
        kin, ins, _, inf = LI.load(subj, trial, case)
        n, win = inf["n"], inf["win"]
        pelvis = pelvis_lab[:n] if case == "camera" else inf["P"]["Pelvis"][:n]
        segs = R.segs0 if case == "camera" else W.imu_inputs(R, subj, trial, inf["tau"])["segs"]
        ref = {s: {k: np.asarray(d["ref"][s][k], float)[:n] for k in ("Fx", "Fy", "copx")} for s in L.SIDES}
        sp_.MASS, sp_.HEIGHT, sp_.FS = mass, height, fs
        segs_ref = {k: {q: (v[:n] if isinstance(v, np.ndarray) else v) for q, v in sv.items()} for k, sv in R.segs0.items()}
        ref_m = sp_.inverse_dynamics(segs_ref, "R", ref["R"]["Fx"], ref["R"]["Fy"], ref["R"]["copx"])
        est, _ = RW.estimates(kin, ins, BW, dev, pelvis, height, fs)
        o = dict(n=n, win=win, fs=fs, BW=BW, mass=mass, ref_fx=ref["R"]["Fx"], ref_hip=ref_m["hip"],
                 onR=ref["R"]["Fy"] > RW.PLATE_CONTACT, onL=ref["L"]["Fy"] > RW.PLATE_CONTACT)
        for m in ("pivot", "graph"):
            fx, fz, cop = est[m]
            o[m + "_fx"] = fx
            o[m + "_hip"] = sp_.inverse_dynamics(segs, "R", fx, fz, cop)["hip"]
        out[case] = o
    return out


def check(S, d, subj, trial):
    for case, o in S.items():
        row = d[(d.case == case) & (d.subject == subj) & (d.trial == trial)].iloc[0]
        st = o["onR"] & o["win"]
        for m in ("pivot", "graph"):
            sh = 100 * RW.rmse(o[m + "_fx"], o["ref_fx"], st) / o["BW"]
            hp = float(np.sqrt(np.mean(((o[m + "_hip"] - o["ref_hip"])[o["win"]] / o["mass"]) ** 2)))
            if abs(sh - row[f"{m}_stance"]) > 1e-9 or abs(hp - row[f"{m}_hip"]) > 1e-9:
                raise SystemExit(f"recomputed trial differs from main.csv: {case} {m} {sh} {row[f'{m}_stance']} "
                                 f"{hp} {row[f'{m}_hip']}")


def window(o, strides=3):
    """Three strides from the first right heel strike (plate) after the middle of the scored window."""
    on = o["onR"]
    hs = np.where(on[1:] & ~on[:-1])[0] + 1
    w = np.where(o["win"])[0]
    mid = (w[0] + w[-1]) // 2
    hs = hs[(hs >= mid) & (hs <= w[-1])]
    return hs[0], hs[strides]


def plot(S, a, b):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(6.9, 4.1), sharex="col", sharey="row")
    rows = []
    for j, (case, title) in enumerate((("camera", "Camera kinematics"), ("imu", "IMU kinematics"))):
        o = S[case]
        t = (np.arange(a, b) - a) / o["fs"]
        ds = (o["onR"] & o["onL"])[a:b]
        for i, (q, scale, ylab) in enumerate((("fx", 100 / o["BW"], "Shear, right foot (% BW)"),
                                               ("hip", 1 / o["mass"], "Hip moment (N m/kg)"))):
            A = ax[i, j]
            edges = np.flatnonzero(np.diff(np.r_[0, ds.astype(int), 0]))
            for e0, e1 in zip(edges[::2], edges[1::2]):
                A.axvspan(t[e0], t[min(e1, len(t) - 1)], color=SHADE, lw=0, zorder=0)
            A.axhline(0, color=GRID, lw=0.6, zorder=1)
            ref = o["ref_" + q][a:b] * scale
            ps, fg = o["pivot_" + q][a:b] * scale, o["graph_" + q][a:b] * scale
            A.plot(t, ref, color=INK, lw=1.4, label="Reference solution", zorder=3)
            A.plot(t, ps, color=C_PS, lw=1.1, ls=(0, (4, 2)), label="PS", zorder=4)
            A.plot(t, fg, color=C_FG, lw=1.2, label="FG", zorder=5)
            for s in ("top", "right"):
                A.spines[s].set_visible(False)
            A.tick_params(length=2.5, width=0.6)
            if j == 0:
                A.set_ylabel(ylab)
            if i == 0:
                A.set_title(title, fontsize=8.5, color=INK)
            if i == 1:
                A.set_xlabel("Time (s)")
            rows.append(pd.DataFrame({"case": case, "quantity": q, "t_s": t, "reference": ref, "PS": ps, "FG": fg,
                                      "double_support": ds}))
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(OUT_DIR, "Fig2_preview.png"), dpi=200)
    return pd.concat(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    f_csv = os.path.join(OUT_DIR, "fig2_series.csv")
    if os.path.exists(f_csv):
        raise SystemExit("exists: " + f_csv)
    subj, trial, d = pick_trial()
    S = series(subj, trial)
    check(S, d, subj, trial)
    a, b = window(S["camera"])
    df = plot(S, a, b)
    df.to_csv(f_csv, index=False)
    with open(os.path.join(OUT_DIR, "fig2_trial.txt"), "x", encoding="utf-8") as fh:
        fh.write(f"trial {subj} {trial}; frames {a}-{b}; speed {wa_speed(trial):.2f} m/s\n")
    print(subj, trial, a, b, f"{wa_speed(trial):.2f} m/s")


def wa_speed(trial):
    return int(trial.split("_")[1]) / 10 / 3.6


if __name__ == "__main__":
    main()
