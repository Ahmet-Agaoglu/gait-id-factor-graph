"""The main result split by walking speed (Online Resource 1, Table S4). Reads results/main/main.csv only (no new solve):
for each kinematic case and speed, the subject means of the factor graph and of the closest competing method of the main
run, and how many subjects have the lower error with the factor graph.
Output: results/by_speed/
Usage: python -I by_speed.py run
"""
import datetime
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_main as RW  # noqa: E402

OUT = os.path.join(config.RESULTS, "by_speed")
F_CSV, F_TXT = os.path.join(OUT, "by_speed.csv"), os.path.join(OUT, "by_speed_summary.txt")
COLS = ("stance", "ds", "ss", "ankle", "knee", "hip")


def speed(trial):
    """Trial name walk_XX: XX tenths of a km/h."""
    return round(int(trial.split("_")[1]) / 10 / 3.6, 2)


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in (F_CSV, F_TXT):
        if os.path.exists(f):
            raise SystemExit("exists: " + f)
    d = pd.read_csv(RW.F_CSV)
    d["speed"] = d.trial.map(speed)
    rows = []
    for case in ("camera", "imu"):
        c = d[d.case == case]
        for v, cv in c.groupby("speed"):
            S = cv.groupby("subject").mean(numeric_only=True)
            for k in COLS:
                comp = "wang" if (case == "imu" and k == "ankle") else "pivot"
                diff = S[f"graph_{k}"] - S[f"{comp}_{k}"]
                rows.append(dict(case=case, speed=v, col=k, comp=comp, n=int(np.isfinite(diff).sum()),
                                 graph=float(S[f"graph_{k}"].mean()), other=float(S[f"{comp}_{k}"].mean()),
                                 pct=float(100 * diff.mean() / S[f"{comp}_{k}"].mean()), n_better=int((diff < 0).sum())))
    r = pd.DataFrame(rows)
    r.to_csv(F_CSV, index=False)
    lines = [f"by_speed.py, {datetime.datetime.now().isoformat(timespec='seconds')}: from main.csv; subject means per "
             "speed; graph / closest competitor (PS; WP at the IMU ankle), change %, subjects lower with the graph."]
    for case in ("camera", "imu"):
        lines.append(f"\n== {case}")
        for v, rv in r[r.case == case].groupby("speed"):
            lines.append(f"{v:4.2f} m/s  " + "  ".join(f"{q.col}: {q.graph:.3f}/{q.other:.3f} {q.pct:+5.1f}% "
                                                     f"({q.n_better}/{q.n})" for q in rv.itertuples()))
    txt = "\n".join(lines)
    print(txt)
    with open(F_TXT, "x", encoding="utf-8") as fh:
        fh.write(txt + "\n")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
