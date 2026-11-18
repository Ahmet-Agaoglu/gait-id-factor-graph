"""Smoothness uncertainty sigma_sm per trial, as the factor graph computes it (factor_graph.smooth_scale: RMS second
difference of the measured insole Fz over frames with three consecutive contacts, both feet pooled), for Table 1. Insoles
only; no force plate channel is read. Both kinematic cases, since the IMU case may use a shorter record.
Output: results/smoothness_scale/
Usage: python -I smoothness_scale.py run
"""
import datetime
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

import factor_graph as L  # noqa: E402
import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

import config  # noqa: E402

OUT = os.path.join(config.RESULTS, "smoothness_scale")
F_CSV, F_TXT = os.path.join(OUT, "smoothness_scale.csv"), os.path.join(OUT, "smoothness_scale_summary.txt")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    rows = []
    for subj, trial in RW.trials():
        for case in ("camera", "imu"):
            kin, ins, BW, info = LI.load(subj, trial, case)
            s = L.smooth_scale(ins, L.INSOLE["F_CONTACT"])
            rows.append(dict(subject=subj, trial=trial, case=case, n=info["n"], BW=BW, sigma_sm_N=s,
                             sigma_sm_pctBW=100 * s / BW))
    d = pd.DataFrame(rows)
    d.to_csv(F_CSV, index=False)
    lines = [f"smoothness_scale.py, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(d)} rows, threshold "
             f"{L.INSOLE['F_CONTACT']:.0f} N."]
    for case in ("camera", "imu"):
        c = d[d.case == case]
        lines.append(f"{case}: trials {len(c)}; sigma_sm median {c.sigma_sm_N.median():.2f} N (IQR "
                     f"{c.sigma_sm_N.quantile(0.25):.2f}-{c.sigma_sm_N.quantile(0.75):.2f}), median "
                     f"{c.sigma_sm_pctBW.median():.3f} %BW (IQR {c.sigma_sm_pctBW.quantile(0.25):.3f}-"
                     f"{c.sigma_sm_pctBW.quantile(0.75):.3f})")
    txt = "\n".join(lines)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
