"""Solve time of the factor graph per trial, camera and IMU cases, one process (Sect. 2.3). The inputs are built first (not
timed); only Graph(...) and solve() are timed. No force plate channel is read. The times depend on the computer.
Output: results/timing/
Usage: python -I timing.py run
"""
import datetime
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import pandas as pd  # noqa: E402

import factor_graph as L  # noqa: E402
import graph_inputs as LI  # noqa: E402
import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "timing")
F_CSV, F_TXT = os.path.join(OUT, "timing.csv"), os.path.join(OUT, "timing_summary.txt")


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    rows = []
    for subj, trial in RW.trials():
        for case, dev in (("camera", L.CAMERA), ("imu", L.IMU)):
            kin, ins, BW, info = LI.load(subj, trial, case)
            t0 = time.perf_counter()
            g = L.Graph(kin, ins, BW, dev)
            g.solve()
            dt = time.perf_counter() - t0
            rows.append(dict(subject=subj, trial=trial, case=case, frames=g.n, unknowns=g.N, it=g.it, seconds=dt))
    d = pd.DataFrame(rows)
    d.to_csv(F_CSV, index=False)
    lines = [f"timing.py, {datetime.datetime.now().isoformat(timespec='seconds')}: {len(d)} solves, one process."]
    for case in ("camera", "imu"):
        c = d[d.case == case]
        lines.append(f"{case}: frames median {c.frames.median():.0f}, unknowns median {c.unknowns.median():.0f}, "
                     f"solve s median {c.seconds.median():.2f} (max {c.seconds.max():.2f}), per frame ms median "
                     f"{1000 * (c.seconds / c.frames).median():.3f}")
    txt = "\n".join(lines)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    main()
