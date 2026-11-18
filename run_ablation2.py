"""Combined-removal arms of the ablation (Sect. 5.5, Table 4). Same scoring as run_ablation.py; only the arms differ. The
last arm keeps only the two balances and the sensors, so the double-support split stays at the start (the NP share): a check
against the newton_* columns of main.csv.
Output: results/ablation2/
Usage: python -I run_ablation2.py run [workers] | summary
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

import factor_graph as L  # noqa: E402
import run_ablation as RA  # noqa: E402
import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "ablation2")
F_CSV, F_TXT, F_LOG = (os.path.join(OUT, f) for f in ("ablation2.csv", "ablation2_summary.txt", "ablation2_log.txt"))
ARMS = (("main", L.FACTORS, {}),
        ("no moment balance, no direction", tuple(f for f in L.FACTORS if f not in ("moment", "direction")), {}),
        ("balances and sensors only", tuple(f for f in L.FACTORS if f not in ("moment", "direction", "smoothness")),
         dict(G_STD=1e-9, D_STD=1e-9)))


def one(args):
    RA.ARMS = ARMS
    return RA.one(args)


def summary():
    if os.path.exists(F_TXT):
        raise SystemExit("exists: " + F_TXT)
    d = pd.read_csv(F_CSV)
    main = pd.read_csv(RW.F_CSV)
    cols = list(RA.COLS) + ["fz_stance"]
    lines = [f"run_ablation2.py summary, {datetime.datetime.now().isoformat(timespec='seconds')}: {d.subject.nunique()} "
             f"subjects, {d.groupby(['subject', 'trial']).ngroups} trials; converged {int(d.converged.sum())}/{len(d)}. "
             "Subject means; change vs main (plus = error grows when the parts are removed); in brackets the number of "
             "subjects where the error grows."]
    for case in ("camera", "imu"):
        chk = d[(d.case == case) & (d.arm == "main")].merge(main[main.case == case], on=["subject", "trial"])
        lines.append(f"check {case}: main vs main.csv max |diff| "
                     f"{max(np.nanmax(np.abs(chk[k] - chk[f'graph_{k}'])) for k in RA.COLS):.2e}")
        c = d[(d.case == case) & (d.arm == "balances and sensors only")].merge(main[main.case == case], on=["subject", "trial"])
        lines.append(f"sanity {case}: 'balances and sensors only' vs newton share, max |diff| per column: " + ", ".join(
            f"{k} {np.nanmax(np.abs(c[k] - c[f'newton_{k}'])):.2e}" for k in cols if f"newton_{k}" in c))
    lines.append("")
    lines.append(f"{'arm':32s} | camera: shear          moments        Fz      | imu: shear          moments        Fz")
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
            parts.append(f"{sh:+6.1f} % ({int((S.stance > S0.stance).sum())}/9)  {mo:+6.1f} % ({int((msub > 1).sum())}/9)  "
                         f"{fz:+6.1f} %")
        lines.append(f"{arm:32s} | " + " | ".join(parts))
    lines.append("")
    lines.append("Detail, subject means per column:")
    for case in ("camera", "imu"):
        g = d[d.case == case].groupby(["arm", "subject"])[cols].mean().groupby("arm").mean()
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
        fh.write(f"run_ablation2.py run, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(rows)} rows, "
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
