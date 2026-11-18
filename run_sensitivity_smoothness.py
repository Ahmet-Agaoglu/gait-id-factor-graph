"""Sensitivity to the smoothness uncertainty sigma_sm (Sect. 5.6, Table 5). The same study as run_sensitivity.py (same
trials, scoring and summary), with one more row: sigma_sm of each trial halved and doubled (factor_graph SM_SCALE 0.5 and
2; sigma_sm itself stays the per-trial value of the method). The main setting is solved again as a check.
Output: results/sensitivity_smoothness/
Usage: python -I run_sensitivity_smoothness.py run [workers] | summary
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import run_sensitivity as RS  # noqa: E402

import config  # noqa: E402

OUT = os.path.join(config.RESULTS, "sensitivity_smoothness")
FILES = tuple(os.path.join(OUT, f) for f in ("sensitivity_smoothness.csv", "sensitivity_smoothness_summary.txt", "sensitivity_smoothness_log.txt"))


def settings():
    return [("main", "-", {}, {}),
            ("smoothness", "low", {}, dict(SM_SCALE=("set", 0.5))),
            ("smoothness", "high", {}, dict(SM_SCALE=("set", 2.0)))]


def patch():
    """Point run_sensitivity at this study. Done only when this file runs (and in its worker processes), never on import, so
    that other modules (tables.ends) still see the main study's settings."""
    RS.OUT = OUT
    RS.F_CSV, RS.F_TXT, RS.F_LOG = FILES
    RS.settings = settings


if __name__ in ("__main__", "__mp_main__"):
    patch()

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "run":
        RS.run(int(sys.argv[2]) if len(sys.argv) > 2 else 6)
    elif sys.argv[1] == "summary":
        RS.summary()
