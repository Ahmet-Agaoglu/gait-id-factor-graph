"""Monte Carlo of the kinematic uncertainties with camera kinematics, plate-free (Online Resource 1, Sect. S2).

The size of every elementary error comes from an independent study (LEVELS, sources in Table S2); its effect on the
whole-body terms comes from this simulation on the data's own markers. No force plate channel enters a value.

Per trial and replica the raw markers or the segment parameters are perturbed, the segment table is rebuilt with the same
filter and segment rules (camera_terms.terms), and the whole-body terms are recomputed. Error = perturbed - nominal over
the scored frames:
  e_x = dSx, e_z = dSz                  horizontal and vertical balance (BW)
  e_m = -dxc Sz + dzc Sx - dHd          moment balance with the forces of the two feet replaced by the whole-body force (m BW)
Each error is split into its trial mean ('c', the constant part) and the rest ('v', the part that varies within the
trial); 't' = RMS of both (t^2 = c^2 + v^2).
Sources, each simulated on its own:
  noise  white noise per marker coordinate
  sta    soft-tissue artifact A u(t) e: A = peak-to-peak / 2.83, u the normalized angle of the joint that moves the marker,
         e a random sagittal unit vector (within 30 deg of cranial for the pelvis markers)
  land   landmark misplacement: one constant sagittal offset per marker and replica, SD per axis
  bsip   segment mass, CoM location and radius of gyration (relative SDs, masses renormalized to the body mass)
  meas   rigid HAT against a HAT on the measured line from the hip to the mean of the C7 and CLAV markers; one
         deterministic difference per trial
The sources are combined in quadrature in each trial; the values of the graph are medians over trials:
SX = t(e_x), SZ = t(e_z), SH = v(e_m), D_STD = c(e_m) (graph_values, summary).
wang_cop() is the CoP placement simulation used by cop_mc.py.
Output: results/camera_mc/camera_mc.csv, camera_mc_summary.txt
Usage: python -I camera_mc.py run [workers] | summary
"""
import copy
import datetime
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import config  # noqa: E402
import planar_model as sp_  # noqa: E402
import factor_graph as F  # noqa: E402

OUT = os.path.join(config.RESULTS, "camera_mc")
F_CSV, F_TXT = (os.path.join(OUT, f) for f in ("camera_mc.csv", "camera_mc_summary.txt"))
SEEDS = os.path.join(HERE, "mc_seeds.csv")
REPS = 30
P2R = 1 / 2.83
SOURCES = ("noise", "sta", "land", "bsip")
QTY = ("x", "z", "m")
GRAPH = (("SX", "x", "t"), ("SZ", "z", "t"), ("SH", "m", "v"), ("D_STD", "m", "c"))

# Inputs (Online Resource 1, Table S2): the source value, or the middle of the source range.
LEVELS = {
    "central": dict(noise_mm=0.23,
                    sta_p2p_mm=dict(asis=13.9, psis=9.4, gtr=24.1, knee=23.4, ankle=9.5, heel=8.5, toe=11.2),
                    land_mm=dict(asis=9.0, psis=9.0, gtr=11.7, knee=4.4, ankle=2.5, heel=6.0, toe=6.0),
                    bsip=dict(hat=(0.075, 0.085, 0.0975), thigh=(0.115, 0.08, 0.095), shank=(0.085, 0.065, 0.085),
                              foot=(0.215, 0.115, 0.105)),
                    cop=dict(shoe_mm=3.7)),
}


def seed(subj, trial, which):
    """Fixed seed of the random number generator for one trial and one simulation ('kinematics' or 'cop')."""
    s = pd.read_csv(SEEDS)
    return int(s.loc[(s.subject == subj) & (s.trial == trial), which].iloc[0])


def stats(e, win):
    """e: (reps, n) errors; RMS over the window: total, constant (per-replica mean), varying."""
    w = e[:, win]
    c = w.mean(axis=1)
    return (float(np.sqrt(np.mean(w ** 2))), float(np.sqrt(np.mean(c ** 2))),
            float(np.sqrt(np.mean((w - c[:, None]) ** 2))))


def perturb_seg(rng, bsip):
    seg = copy.deepcopy(sp_.SEG)
    for name in seg:
        s_m, s_c, s_rg = bsip[name]
        seg[name]["m"] *= 1.0 + rng.normal(0.0, s_m)
        seg[name]["c"] *= 1.0 + rng.normal(0.0, s_c)
        seg[name]["rg"] *= 1.0 + rng.normal(0.0, s_rg)
    tot = 2 * (seg["foot"]["m"] + seg["shank"]["m"] + seg["thigh"]["m"]) + seg["hat"]["m"]
    for name in seg:
        seg[name]["m"] /= tot / 1.0
    return seg


def z(x):
    return (x - x.mean()) / max(x.std(), 1e-12)


def drives(P, n):
    """Normalized joint angles that drive the soft-tissue artifact of each marker class."""
    def ang(a, b):
        v = P[b][:n] - P[a][:n]
        return np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    out = {}
    for s in ("R", "L"):
        th, sh, ft = ang(s + ".Hip", s + ".Knee"), ang(s + ".Knee", s + ".Ankle"), ang(s + ".Heel", s + ".MT5")
        out["hip" + s], out["knee" + s], out["ankle" + s] = z(th), z(th - sh), z(sh - ft)
    out["pelvis"] = z(out["hipR"] + out["hipL"])
    return out


DRIVER = dict(asis="hip", gtr="hip", knee="knee", ankle="knee", heel="ankle", toe="ankle")


def driver(cls, side, dr):
    return dr["pelvis"] if cls == "psis" else dr[DRIVER[cls] + side]


def errors(Kp, Kt, BW):
    """computed (Kp) - truth (Kt) for the balance quantities."""
    dxc, dzc = Kp["xc"] - Kt["xc"], Kp["zc"] - Kt["zc"]
    return dict(x=(Kp["Sx"] - Kt["Sx"]) / BW, z=(Kp["Sz"] - Kt["Sz"]) / BW,
                m=(-dxc * Kt["Sz"] + dzc * Kt["Sx"] - (Kp["Hd"] - Kt["Hd"])) / BW)


class Wang:
    """One trial of Wang et al. (2023): raw markers, camera segment table, insole channels."""
    CLS = dict(RASI="asis", LASI="asis", SACR="psis", RLFE="knee", LLFE="knee", RLM="ankle", LLM="ankle",
               RCAL="heel", LCAL="heel", R5MT="toe", L5MT="toe")
    FS = 100.0

    def __init__(self, subj, trial):
        import camera_terms as KB
        import wang_data as wa
        self.KB = KB
        d = wa.load_trial(subj, trial)
        info = wa.subject_info()[subj]
        _, self.raw = wa.read_trc(d["paths"]["trc"])
        self.n = n = min(len(d["t"]), len(next(iter(self.raw.values()))))
        self.mass, self.height = float(info["mass"]), float(info["height"])
        _, self.segs0, self.P0 = KB.terms(self.raw, n, self.mass, self.height)
        self.win = np.zeros(n, bool)
        self.win[int(self.FS):n - int(self.FS)] = True
        self.ins = {s: {k: np.asarray(d["insole"][s][k], float)[:n] for k in ("Fy", "copx")} for s in ("R", "L")}

    def side(self, k):
        return k[0]

    def build(self, raw, seg=None):
        return self.KB.terms(raw, self.n, self.mass, self.height, seg)[1]

    def planar(self, raw):
        return self.KB.terms(raw, self.n, self.mass, self.height)[2]

    def perturb(self, disp, noise_m, rng):
        out = dict(self.raw)
        for k in self.CLS:
            a = np.array(self.raw[k], float, copy=True)
            if k in disp:
                a[:self.n, :2] += disp[k]
            if noise_m:
                a[:self.n, :2] += rng.normal(0.0, noise_m, size=(self.n, 2))
            out[k] = a
        return out


def displacements(R, rng, prm, sta, land, dr):
    n = R.n
    disp = {}
    for k, cls in R.CLS.items():
        d = np.zeros((n, 2))
        if sta:
            pel = cls in ("asis", "psis")
            phi = np.pi / 2 + rng.uniform(-np.pi / 6, np.pi / 6) if pel else rng.uniform(0, 2 * np.pi)
            d += (prm["sta_p2p_mm"][cls] * P2R / 1000.0) * driver(cls, R.side(k), dr)[:, None] * \
                np.array([np.cos(phi), np.sin(phi)])[None, :]
        if land:
            d += rng.normal(0.0, prm["land_mm"][cls] / 1000.0, size=2)[None, :]
        disp[k] = d
    return disp


def run_trial(args):
    subj, trial, level, reps = args
    import camera_terms as KB
    prm = LEVELS[level]
    t0 = time.time()
    R = Wang(subj, trial)
    n, fs, BW = R.n, R.FS, R.mass * F.G
    rng = np.random.default_rng(seed(subj, trial, "kinematics"))
    K0 = F.whole_body_terms(R.segs0)
    dr = drives(R.P0, n)
    row = dict(subject=subj, trial=trial, n=n, n_win=int(R.win.sum()), mass=R.mass)
    for src in SOURCES:
        E = {q: np.zeros((reps, n)) for q in QTY}
        for r in range(reps):
            seg = perturb_seg(rng, prm["bsip"]) if src == "bsip" else None
            if src in ("noise", "sta", "land"):
                disp = displacements(R, rng, prm, src == "sta", src == "land", dr)
                raw = R.perturb(disp, prm["noise_mm"] / 1000.0 if src == "noise" else 0.0, rng)
                Kp = F.whole_body_terms(R.build(raw, seg))
            else:
                Kp = F.whole_body_terms(R.build(R.raw, seg))
            e = errors(Kp, K0, BW)
            for q in QTY:
                E[q][r] = e[q]
        for q in QTY:
            row[f"{q}_{src}_t"], row[f"{q}_{src}_c"], row[f"{q}_{src}_v"] = stats(E[q], R.win)

    def mk_(k):
        a = pd.DataFrame(np.column_stack([R.raw[k][:n, 0], R.raw[k][:n, 1]])).interpolate(limit_direction="both")
        return sp_.lowpass(a.values, 6.0, fs)
    s2 = KB.hat_on_line(R.segs0, R.segs0[("B", "hat")]["prox"], 0.5 * (mk_("C7") + mk_("CLAV")), R.mass)
    e = errors(K0, F.whole_body_terms(s2), BW)
    for q in QTY:
        row[f"{q}_meas_t"], row[f"{q}_meas_c"], row[f"{q}_meas_v"] = stats(e[q][None, :], R.win)
    row["sec"] = time.time() - t0
    return row


def wang_cop(R, rng, prm, reps, thr):
    """Insole CoP placed in the lab frame on the heel -> MT5 marker line (wang_segments.insole_cop_to_lab, as the graph's
    camera input). Errors (m) over the insole contact frames (insole force > thr) of both feet inside the window."""
    import wang_segments as wr
    n, fs = R.n, R.FS
    t = np.arange(n) / fs
    on = {s: (R.ins[s]["Fy"] > thr) & R.win for s in ("R", "L")}

    def place(P, s, cop):
        return wr.insole_cop_to_lab(P, s, cop, n, heel_key=s + ".Heel")
    base = {s: place(R.P0, s, R.ins[s]["copx"]) for s in ("R", "L")}
    out = {}
    for src in ("cop_mark", "cop_shoe", "cop_time", "cop_all"):
        tot, con, var = [], [], []
        for _ in range(reps):
            P = R.P0
            if src in ("cop_mark", "cop_all"):
                disp = {k: rng.normal(0.0, prm["land_mm"][R.CLS[k]] / 1000.0, size=2)[None, :] * np.ones((n, 1))
                        for k in ("RCAL", "LCAL", "R5MT", "L5MT")}
                P = R.planar(R.perturb(disp, prm["noise_mm"] / 1000.0, rng))
            for s in ("R", "L"):
                cop = R.ins[s]["copx"]
                if src in ("cop_time", "cop_all"):
                    cop = np.interp(t - rng.normal(0.0, prm["cop"]["dt_ms"] / 1000.0), t, cop)
                if src in ("cop_shoe", "cop_all"):
                    cop = cop + rng.normal(0.0, prm["cop"]["shoe_mm"] / 1000.0)
                e = (place(P, s, cop) - base[s])[on[s]]
                c = e.mean()
                tot.append(np.mean(e ** 2))
                con.append(c ** 2)
                var.append(np.mean((e - c) ** 2))
        out[f"{src}_t"], out[f"{src}_c"], out[f"{src}_v"] = (float(np.sqrt(np.mean(a))) for a in (tot, con, var))
    return out


def run(workers=10):
    import trials as TR
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(F_CSV):
        raise SystemExit("exists: " + F_CSV)
    J = [(s, t, "central", REPS) for s, t in TR.trials()]
    t0, rows, bad = time.time(), [], []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(run_trial, j): j for j in J}
        for i, fu in enumerate(futs, 1):
            try:
                rows.append(fu.result())
            except Exception as ex:      # logged, never silent
                bad.append((futs[fu], repr(ex)))
            if i % 25 == 0:
                print(f"{i}/{len(J)} {time.time() - t0:.0f} s", flush=True)
    pd.DataFrame(rows).to_csv(F_CSV, index=False)
    print(f"done {len(rows)} trials, {len(bad)} errors, {time.time() - t0:.0f} s")
    for b in bad:
        print("ERROR", b)
    summary()


def graph_values(d):
    """Per trial: the five sources in quadrature, for each value of the graph."""
    out = pd.DataFrame(dict(subject=d.subject, trial=d.trial))
    for name, q, k in GRAPH:
        out[name] = np.sqrt(sum(d[f"{q}_{src}_{k}"] ** 2 for src in SOURCES + ("meas",)))
    return out


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    v = graph_values(d)
    out = [f"camera_mc.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(d)} trials, "
           f"{d.subject.nunique()} subjects, {REPS} replicas per source. Inputs: {LEVELS['central']!r}",
           "Medians over trials (x, z: BW; m: m BW). t total, c constant within the trial, v varying.",
           pd.DataFrame({src: [d[f"{q}_{src}_{k}"].median() for q in QTY for k in "tcv"] for src in SOURCES + ("meas",)},
                        index=[f"{q}_{k}" for q in QTY for k in "tcv"]).round(4).to_string(),
           "", "Values of the graph (median over trials of the five sources in quadrature; range of subject medians):"]
    for name, q, k in GRAPH:
        s = v.groupby("subject")[name].median()
        out.append(f"{name:6s} {v[name].median():.4f}  ({s.min():.4f}-{s.max():.4f})")
    txt = "\n".join(out)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    elif sys.argv[1] == "summary":
        summary()
