"""The trials analysed: every walking trial of the nine subjects of Wang et al. (2023) whose marker data are complete
(every marker of the planar model missing in at most 5 % of the frames, wang_segments.planar_from_markers); 53 of 54.
Output: results/trials.csv
Usage: python -I trials.py run
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

import config  # noqa: E402

F_CSV = os.path.join(config.RESULTS, "trials.csv")


def trials():
    """[(subject, trial), ...] in the order of results/trials.csv."""
    return [tuple(x) for x in pd.read_csv(F_CSV)[["subject", "trial"]].values]


def run():
    import wang_data as wa
    import wang_segments as wr
    if os.path.exists(F_CSV):
        raise SystemExit("exists: " + F_CSV)
    rows = []
    for subj in sorted(s for s in wa.subject_info() if os.path.isdir(os.path.join(wa.ROOT, s))):   # released: Subj04-12
        for trial in wa.WALKS:
            d = wa.load_trial(subj, trial)
            trc = wa.read_trc(d["paths"]["trc"])
            n = min(len(d["t"]), len(trc[0]))
            ok = bool(wr.planar_from_markers(trc, n)[2])
            rows.append(dict(subject=subj, trial=trial, speed=round(wa.SPEED[trial], 3), markers_complete=ok))
    d = pd.DataFrame(rows)
    os.makedirs(config.RESULTS, exist_ok=True)
    d[d.markers_complete][["subject", "trial", "speed"]].to_csv(F_CSV, index=False)
    print(f"{int(d.markers_complete.sum())} of {len(d)} trials kept; left out: "
          + ", ".join(d.loc[~d.markers_complete, "subject"] + " " + d.loc[~d.markers_complete, "trial"]))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run()
