"""Run every step of the study in order, from the data to the tables and Fig. 2.

Each step is a separate script (see README.md); this file only calls them one after the other with the same Python
(isolated mode, -I) and stops at the first failure. Steps that solve many trials run in parallel processes. No step
overwrites an existing result file: to recompute, set GAIT_FG_RESULTS to an empty folder (README.md).
Usage: python -I run_all.py [workers] [first step]     (workers default 6; first step e.g. run_main.py)
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
W = sys.argv[1] if len(sys.argv) > 1 else "6"

STEPS = [
    ("trials.py", "run"),                         # trial list (markers complete)
    ("time_shift.py", "run", W),                  # insole vs camera time shift (Online Resource 1)
    ("camera_mc.py", "run", W),                   # camera kinematic uncertainties (Monte Carlo)
    ("cop_mc.py", "run", W),                      # camera CoP placement uncertainty (Monte Carlo)
    ("imu_budget.py", "run", W),                  # IMU uncertainties (IMU - camera)
    ("smoothness_ratio.py", "run", W),            # horizontal-to-vertical smoothness ratio
    ("contact_threshold.py", "run", W),           # insole contact threshold
    ("smoothness_scale.py", "run"),               # per-trial smoothness uncertainty (Table 1)
    ("run_main.py", "run", W),                    # main run (Table 3)
    ("run_ablation.py", "run", W),                # ablation (Table 4)
    ("run_ablation2.py", "run", W),               # combined-removal arms (Table 4)
    ("run_sensitivity.py", "run", W),             # sensitivity (Table 5)
    ("run_sensitivity_smoothness.py", "run", W),  # sensitivity to sigma_sm (Table 5)
    ("direction_factor.py", "run", W),            # where the direction factor acts
    ("planar_vs_lab.py", "run", W),               # planar reference vs laboratory inverse dynamics
    ("hip_center.py", "run", W),                  # effect of the hip-center rule
    ("timing.py", "run"),                         # solve times (machine-dependent)
    ("by_speed.py", "run"),                       # main result by speed
    ("figure2_data.py",),                         # Fig. 2 series
    ("figure2.py",),                              # Fig. 2
    ("paper_tables.py",),                         # LaTeX tables
]

if __name__ == "__main__":
    t0 = time.time()
    first = [s[0] for s in STEPS].index(sys.argv[2]) if len(sys.argv) > 2 else 0
    for step in STEPS[first:]:
        t = time.time()
        print(f"== {' '.join(step)}", flush=True)
        subprocess.run([sys.executable, "-I", os.path.join(HERE, step[0]), *step[1:]], check=True)
        print(f"   done in {time.time() - t:.0f} s", flush=True)
    print(f"all steps done in {(time.time() - t0) / 60:.1f} min")
