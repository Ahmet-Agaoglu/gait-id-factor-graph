"""Factor graph for the ground reaction forces of walking without shear force measurement (Sect. 2 of the article).

Planar (sagittal) model on level ground: the CoP lies at height 0 and the free moment is neglected. Inputs of one trial:
the whole-body kinematic terms of a segment table (whole_body_terms: CoM position xc, zc; force sums Sx, Sz; rate of
change of the angular momentum about the CoM, Hd) and, for each foot, the measured vertical force Fz and CoP copx in the
same frame. The walking direction is +x. No force plate value enters this file.

Unknowns: Fx, Fz and CoP x of each foot in every frame where that foot's measured Fz exceeds the contact threshold;
the scale g of each foot and the moment offset delta, once per trial. Out of contact a foot carries no force.

Seven parts (residual / standard deviation), plus the two per-trial constants:
  1 horizontal balance   FxR + FxL - Sx                                          / SX BW    frames with any contact
  2 vertical balance     FzR + FzL - Sz                                          / SZ BW    frames with any contact
  3 moment balance       sum_s (x_s - xc) Fz_s + zc (FxR + FxL) - (Hd - delta)   / SH BW    frames with any contact
  4 direction            Fx_s - (px - x_s) / pz Fz_s                             / SDIR BW  double support only
                         pivot (px, pz) = (xc + PIVOT_FWD, zc + PIVOT_UP)
  5 sensor Fz            g_s Fz_s - Fz measured                                  / SI BW    contact frames of foot s
  6 sensor CoP           x_s - CoP measured                                      / SCOP     contact frames of foot s
  7 force smoothness     second differences of Fz_s / d2fz and of Fx_s / (FX_RATIO d2fz), whole trial
     d2fz = RMS second difference of the measured Fz over frames with three consecutive contacts (both feet)
  constants              (g_s - 1) / G_STD, delta / (D_STD BW)
Solved by Levenberg-Marquardt on the sparse normal equations (frame-major columns, banded normal matrix).

MODEL holds the model-dependent values (both kinematic cases); CAMERA and IMU the measurement-device values of Wang et al.
2023 (camera markers or Xsens suit, with Moticon insoles). Values and sources: Table 2 of the article.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

G = 9.81
SIDES = ("R", "L")
SCALARS = ("gR", "gL", "delta")
MODEL = dict(PIVOT_UP=0.0,     # m above the CoM, double support (Vielemeyer et al. 2021)
             PIVOT_FWD=0.051,  # m ahead of the CoM, double support (Vielemeyer et al. 2021)
             SDIR=0.03,        # BW, direction deviation (declared)
             FX_RATIO=0.55)    # horizontal / vertical second-difference ratio (Wang single support, kinematics)
# insole (both cases): contact threshold from the insole's agreement with kinematic gait events (70 N), Fz 10.5 % BW
# (Zelik 2026), scale prior 1 +- 0.03 (Cramer 2022); kinematics and CoP placement per case (Monte Carlo / IMU - camera)
INSOLE = dict(F_CONTACT=70.0, SI=0.105, G_STD=0.03)
CAMERA = dict(INSOLE, SX=0.037, SZ=0.024, SH=0.018, D_STD=0.015, SCOP=0.017)
IMU = dict(INSOLE, SX=0.069, SZ=0.045, SH=0.034, D_STD=0.013, SCOP=0.046)
FACTORS = ("horizontal", "vertical", "moment", "direction", "sensor_fz", "sensor_cop", "smoothness", "constants")
MAX_ITER, TOL, LAM0 = 60, 1e-10, 1e-6


def smooth_scale(ins, thr):
    """RMS second difference of the measured Fz over frames with three consecutive contacts, both feet [N]."""
    out = []
    for s in SIDES:
        f = np.asarray(ins[s]["Fy"], float)
        on = f > thr
        t = np.where(on[:-2] & on[1:-1] & on[2:])[0]
        out.append(f[t] - 2 * f[t + 1] + f[t + 2])
    return float(np.sqrt(np.mean(np.concatenate(out) ** 2)))


class Graph:
    def __init__(self, kin, ins, BW, device, model=MODEL, factors=FACTORS):
        self.p = dict(model, **device)
        self.use = tuple(factors)
        self.kin = {k: np.asarray(kin[k], float) for k in ("xc", "zc", "Sx", "Sz", "Hd")}
        self.ins = {s: {k: np.asarray(ins[s][k], float) for k in ("Fy", "copx")} for s in SIDES}
        self.BW = float(BW)
        self.n = n = len(self.kin["Sx"])
        self.on = {s: self.ins[s]["Fy"] > self.p["F_CONTACT"] for s in SIDES}
        for v in self.kin.values():
            if len(v) != n or not np.all(np.isfinite(v)):
                raise ValueError("kinematic terms must be finite and of one length")
        for s in SIDES:
            if len(self.on[s]) != n or not np.all(np.isfinite(self.ins[s]["copx"][self.on[s]])):
                raise ValueError("insole must be of the kinematic length, with a finite CoP in every contact frame")
        self.any = self.on["R"] | self.on["L"]
        self.ds = self.on["R"] & self.on["L"]
        # SM_SCALE: factor on sigma_sm, used only by the sensitivity study (absent = 1, the method)
        self.d2fz = smooth_scale(self.ins, self.p["F_CONTACT"]) * self.p.get("SM_SCALE", 1.0)
        self.d2fx = self.p["FX_RATIO"] * self.d2fz
        fam = [(q, s) for s in SIDES for q in ("Fx", "Fz", "x")]
        free = np.column_stack([np.repeat(self.on[s][:, None], 3, axis=1) for s in SIDES])
        cm = np.full(free.shape, -1)
        cm[free] = np.arange(int(free.sum()))
        self.cols = {f: cm[:, j].copy() for j, f in enumerate(fam)}
        k = int(free.sum())
        for name in SCALARS:
            self.cols[name] = k
            k += 1
        self.N = k

    def full(self, v, q, s):
        """Series of one unknown over the whole trial; 0 out of contact."""
        cm = self.cols[(q, s)]
        out = np.zeros(self.n)
        free = cm >= 0
        out[free] = v[cm[free]]
        return out

    def start(self):
        """Measured Fz and CoP at face value; Sx shared between the feet in proportion to the measured Fz."""
        v = np.zeros(self.N)
        v[self.cols["gR"]] = v[self.cols["gL"]] = 1.0
        fz = {s: np.where(self.on[s], self.ins[s]["Fy"], 0.0) for s in SIDES}
        tot = np.maximum(fz["R"] + fz["L"], 1e-9)
        for s in SIDES:
            cm = self.cols[("Fz", s)]
            fr = cm >= 0
            v[cm[fr]] = fz[s][fr]
            v[self.cols[("x", s)][fr]] = self.ins[s]["copx"][fr]
            v[self.cols[("Fx", s)][fr]] = (self.kin["Sx"] * fz[s] / tot)[fr]
        return v

    def eval(self, v, jac=True, parts=False):
        """Weighted residuals r and their Jacobian J (CSR); parts=True also returns the row ranges of each part."""
        p, BW, n, kin, use = self.p, self.BW, self.n, self.kin, self.use
        F = {(q, s): self.full(v, q, s) for q in ("Fx", "Fz", "x") for s in SIDES}
        sc = {name: v[self.cols[name]] for name in SCALARS}
        R, rows, cols, vals, rng = [], [], [], [], {}
        nrow = [0]

        def block(name, r, terms):
            base = nrow[0]
            R.append(r)
            rng.setdefault(name, []).append((base, base + len(r)))
            if jac:
                idx = np.arange(len(r))
                for key, frames, coef in terms:
                    coef = np.broadcast_to(np.asarray(coef, float), r.shape)
                    if isinstance(key, str):
                        rows.append(base + idx)
                        cols.append(np.full(len(r), self.cols[key]))
                        vals.append(coef)
                    else:
                        cm = self.cols[key][frames]
                        m = cm >= 0
                        rows.append(base + idx[m])
                        cols.append(cm[m])
                        vals.append(coef[m])
            nrow[0] += len(r)

        A = np.where(self.any)[0]
        if "horizontal" in use:
            w = 1.0 / (p["SX"] * BW)
            block("horizontal", w * (F[("Fx", "R")] + F[("Fx", "L")] - kin["Sx"])[A], [(("Fx", s), A, w) for s in SIDES])
        if "vertical" in use:
            w = 1.0 / (p["SZ"] * BW)
            block("vertical", w * (F[("Fz", "R")] + F[("Fz", "L")] - kin["Sz"])[A], [(("Fz", s), A, w) for s in SIDES])
        if "moment" in use:
            w = 1.0 / (p["SH"] * BW)
            xc, zc = kin["xc"][A], kin["zc"][A]
            m = sum((F[("x", s)][A] - xc) * F[("Fz", s)][A] for s in SIDES) + zc * (F[("Fx", "R")] + F[("Fx", "L")])[A]
            terms = [("delta", A, w)]
            for s in SIDES:
                terms += [(("Fx", s), A, w * zc), (("Fz", s), A, w * (F[("x", s)][A] - xc)),
                          (("x", s), A, w * F[("Fz", s)][A])]
            block("moment", w * (m - kin["Hd"][A] + sc["delta"]), terms)
        D = np.where(self.ds)[0]
        if "direction" in use and len(D):
            w = 1.0 / (p["SDIR"] * BW)
            px, pz = kin["xc"][D] + p["PIVOT_FWD"], kin["zc"][D] + p["PIVOT_UP"]
            for s in SIDES:
                fx, fz, x = F[("Fx", s)][D], F[("Fz", s)][D], F[("x", s)][D]
                th = (px - x) / pz
                block("direction", w * (fx - th * fz), [(("Fx", s), D, w), (("Fz", s), D, -w * th), (("x", s), D, w * fz / pz)])
        for s in SIDES:
            fr = np.where(self.on[s])[0]
            g = sc["g" + s]
            if "sensor_fz" in use:
                w = 1.0 / (p["SI"] * BW)
                fz = F[("Fz", s)][fr]
                block("sensor_fz", w * (g * fz - self.ins[s]["Fy"][fr]), [(("Fz", s), fr, w * g), ("g" + s, fr, w * fz)])
            if "sensor_cop" in use:
                w = 1.0 / p["SCOP"]
                block("sensor_cop", w * (F[("x", s)][fr] - self.ins[s]["copx"][fr]), [(("x", s), fr, w)])
            if "smoothness" in use:
                t = np.arange(n - 2)
                for q, sd in (("Fx", self.d2fx), ("Fz", self.d2fz)):
                    f = F[(q, s)]
                    w = 1.0 / sd
                    block("smoothness", w * (f[t] - 2 * f[t + 1] + f[t + 2]),
                          [((q, s), t, w), ((q, s), t + 1, -2 * w), ((q, s), t + 2, w)])
        if "constants" in use:
            for s in SIDES:
                block("constants", np.array([(sc["g" + s] - 1.0) / p["G_STD"]]), [("g" + s, None, 1.0 / p["G_STD"])])
            sd = p["D_STD"] * BW
            block("constants", np.array([sc["delta"] / sd]), [("delta", None, 1.0 / sd)])
        r = np.concatenate(R)
        if not jac:
            return (r, rng) if parts else r
        J = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(r), self.N))
        return (r, J, rng) if parts else (r, J)

    def solve(self):
        v = self.start()
        r, J = self.eval(v)
        cost = float(r @ r)
        lam, converged, it = LAM0, False, 0
        for it in range(1, MAX_ITER + 1):
            A = (J.T @ J).tocsc()
            b = J.T @ r
            d = A.diagonal() + 1e-12
            while True:
                lu = splu((A + sp.diags(lam * d)).tocsc(), permc_spec="NATURAL", diag_pivot_thresh=0.0,
                          options=dict(SymmetricMode=True))
                dv = -lu.solve(b)
                c2 = float(np.sum(self.eval(v + dv, jac=False) ** 2))
                if c2 <= cost or lam > 1e10:
                    break
                lam *= 10.0
            if c2 > cost:
                break
            v = v + dv
            converged = (cost - c2) <= TOL * cost
            r, J = self.eval(v)
            cost = c2
            lam = max(lam / 10.0, 1e-12)
            if converged:
                break
        self.v, self.cost, self.it, self.converged = v, cost, it, converged
        out = {s: dict(Fx=self.full(v, "Fx", s), Fz=self.full(v, "Fz", s),
                       copx=np.where(self.on[s], self.full(v, "x", s), self.ins[s]["copx"])) for s in SIDES}
        out.update({name: float(v[self.cols[name]]) for name in SCALARS})
        return out


def whole_body_terms(segs):
    """CoM position (xc, zc), kinematic force sums Sx, Sz and dH/dt about the CoM, from a segment table
    (mass m, inertia I, COM com and its acceleration acom as n x 2 arrays, angular acceleration alpha)."""
    M = sum(v["m"] for v in segs.values())
    rc = sum(v["m"] * v["com"] for v in segs.values()) / M
    ac = sum(v["m"] * v["acom"] for v in segs.values()) / M

    def cross2(r, f):
        return r[:, 0] * f[:, 1] - r[:, 1] * f[:, 0]
    Hd = sum(v["I"] * v["alpha"] + v["m"] * cross2(v["com"] - rc, v["acom"] - ac) for v in segs.values())
    Sx = sum(v["m"] * v["acom"][:, 0] for v in segs.values())
    Sz = sum(v["m"] * (v["acom"][:, 1] + G) for v in segs.values())
    return dict(xc=rc[:, 0], zc=rc[:, 1], Sx=Sx, Sz=Sz, Hd=Hd)
