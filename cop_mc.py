"""Monte Carlo of the insole CoP placement with camera kinematics (Online Resource 1, Table S2), plate-free.
Same code as camera_mc.wang_cop: noise and misplacement of the heel and fifth-metatarsal markers, insole displacement in the
shoe, insole-marker time shift. The time-shift SD is the value measured in the data (time_shift.py: RMS over the reliable
trials). The errors are taken over the insole contact frames (contact threshold of the graph). The total CoP uncertainty is
the sensor's 15 mm and the placement in quadrature (paper_numbers.py). No force plate channel is used.
Output: results/cop_mc/cop_mc.csv, cop_mc_summary.txt
Usage: python -I cop_mc.py run [workers]
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
import camera_mc as K  # noqa: E402
import factor_graph as L  # noqa: E402

DT_MS = 12.3
SENSOR_MM = 15.0
THR = L.INSOLE["F_CONTACT"]
OUT = os.path.join(config.RESULTS, "cop_mc")
F_CSV, F_TXT = (os.path.join(OUT, f) for f in ("cop_mc.csv", "cop_mc_summary.txt"))


def one(args):
    subj, trial, level = args
    prm = copy.deepcopy(K.LEVELS[level])
    prm["cop"]["dt_ms"] = DT_MS
    R = K.Wang(subj, trial)
    rng = np.random.default_rng(K.seed(subj, trial, "cop"))
    return dict(subject=subj, trial=trial, level=level, **K.wang_cop(R, rng, prm, K.REPS, THR))


def run(workers=10):
    import trials as TR
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        d = pd.DataFrame(list(pool.map(one, [(s, t, "central") for s, t in TR.trials()])))
    d.to_csv(F_CSV, index=False)
    parts = ", ".join(f"{s} t {1000 * d[s + '_t'].median():.1f} c {1000 * d[s + '_c'].median():.1f} "
                      f"v {1000 * d[s + '_v'].median():.1f}" for s in ("cop_mark", "cop_shoe", "cop_time", "cop_all"))
    tot = np.sqrt(SENSOR_MM ** 2 + (1000 * d["cop_all_t"]) ** 2)
    txt = (f"cop_mc.py, {datetime.datetime.now().isoformat(timespec='seconds')}: time shift SD {DT_MS} ms (measured), "
           f"{K.REPS} replicas, contact threshold {THR:.0f} N, {time.time() - t0:.0f} s. Medians over trials, mm.\n"
           f"{d.subject.nunique()} subjects, {len(d)} trials: {parts}; total with the sensor {SENSOR_MM:.0f} mm: "
           f"median {tot.median():.1f} mm")
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
