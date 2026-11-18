"""Fig. 2 drawn from the saved series (results/figure2/fig2_series.csv, written by figure2_data.py) without recomputing
anything: drawn at 6.9 x 4.1 in and scaled to the text width when printed.
Output: output/Fig2.pdf
Usage: python -I figure2.py
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402

SERIES = os.path.join(config.RESULTS, "figure2", "fig2_series.csv")
OUT_PDF = os.path.join(config.OUTPUT, "Fig2.pdf")
INK, INK2, GRID, SHADE = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"   # as figure2_data.py
C_FG, C_PS = "#2a78d6", "#eb6834"


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    d = pd.read_csv(SERIES)
    plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, ax = plt.subplots(2, 2, figsize=(6.9, 4.1), sharex="col", sharey="row")
    for j, (case, title) in enumerate((("camera", "Camera kinematics"), ("imu", "IMU kinematics"))):
        for i, (q, ylab) in enumerate((("fx", "Shear, right foot (% BW)"), ("hip", "Hip moment (N m/kg)"))):
            s = d[(d.case == case) & (d.quantity == q)]
            t, ds = s.t_s.values, s.double_support.values.astype(bool)
            A = ax[i, j]
            edges = np.flatnonzero(np.diff(np.r_[0, ds.astype(int), 0]))
            for e0, e1 in zip(edges[::2], edges[1::2]):
                A.axvspan(t[e0], t[min(e1, len(t) - 1)], color=SHADE, lw=0, zorder=0)
            A.axhline(0, color=GRID, lw=0.6, zorder=1)
            A.plot(t, s.reference.values, color=INK, lw=1.4, label="Reference solution", zorder=3)
            A.plot(t, s.PS.values, color=C_PS, lw=1.1, ls=(0, (4, 2)), label="PS", zorder=4)
            A.plot(t, s.FG.values, color=C_FG, lw=1.2, label="FG", zorder=5)
            for sp in ("top", "right"):
                A.spines[sp].set_visible(False)
            A.tick_params(length=2.5, width=0.6)
            if j == 0:
                A.set_ylabel(ylab)
            if i == 0:
                A.set_title(title, fontsize=8.5, color=INK)
            if i == 1:
                A.set_xlabel("Time (s)")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(config.OUTPUT, exist_ok=True)
    fig.savefig(OUT_PDF)
    print("written", OUT_PDF)


if __name__ == "__main__":
    main()
