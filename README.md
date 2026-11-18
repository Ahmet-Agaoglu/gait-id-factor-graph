# Gait inverse dynamics without shear force measurement via factor graph optimization

Code of the article *Gait Inverse Dynamics Without Shear Force Measurement via Factor Graph Optimization*
(A. Agaoglu). Pressure insoles and similar devices measure each foot's vertical force and center of pressure (CoP), but
not the horizontal force (shear). A factor graph, one sparse nonlinear least-squares problem over a whole trial, estimates
each foot's shear and corrects its measured vertical force; the joint moments of the leg then follow by inverse dynamics.
Its factors are the whole-body balances on the measured kinematics, the measured vertical force and CoP, a force-direction
regularity of double support and the smoothness of the forces, each weighted by its own uncertainty.

The code reproduces every result, table and figure of the article from the public data of Wang et al. (2023), and checks
every number printed in the article against the saved results. The force plates of the data set are used only to score the
estimates, never inside the method.

## Requirements

* Python 3.14 with the packages in `requirements.txt` (`pip install -r requirements.txt`), the versions the results were
  produced with.
* An extractor for `.rar` archives (7-Zip, WinRAR or `unrar`) for the data.

## Data

Wang et al. (2023): nine subjects walking on an instrumented treadmill at six speeds, with pressure insoles, an IMU suit
(Xsens) and optical motion capture. License: CC BY 3.0 NL. The data are not redistributed here.

* Record: Zenodo, <https://doi.org/10.5281/zenodo.6457662>
* Download `Processed_data.rar` (17.9 GB) and extract it so that the subject folders sit in
  `data/Wang2023/Processed_data/`:

```
data/
  Wang2023/
    Processed_data/        Subj04 ... Subj12, subjs_info.txt (extracted from Processed_data.rar)
    cache/                 written by wang_data.py on first use
```

Other locations are set with environment variables (`config.py`):

| Variable | Default | Content |
|---|---|---|
| `GAIT_FG_WANG` | `./data/Wang2023` | the data set (above) |
| `GAIT_FG_RESULTS` | `./results` | result files |
| `GAIT_FG_OUTPUT` | `./output` | LaTeX tables and Fig. 2 |

## Running

All scripts are run in isolated mode (`python -I`); each one states its usage in its first lines.

```
python -I run_all.py 10                  # every step in order, 10 worker processes
python -I run_all.py 10 run_main.py      # from one step on
python -I verify.py <article folder>     # the check of the article's numbers alone, in seconds
```

With 10 worker processes the steps take about 8 minutes on a laptop (Intel Core i7-13700H), after the first reading of
the data set into `cache/` (a few minutes, once).

No script overwrites an existing result file. `results/` holds the files the article draws on; to recompute them, set
`GAIT_FG_RESULTS` to an empty folder and run `run_all.py`. The Monte Carlo steps use fixed seeds (`mc_seeds.csv`), so
the rerun gives the same numbers; only the solve times (`timing.py`) depend on the computer.

`verify.py` reruns `paper_tables.py` in memory and compares its output with the table files of the article, recomputes
each number printed in the text from `results/` and compares it with the printed value, rounded the same way, and lists
any number of the text that no check covers. It exits with a non-zero status on any mismatch. The article source
(`main.tex`, `main.aux`, `esm1.tex` and the table files) is not part of this repository; its folder is the argument.

## Files

Method:

| File | Content |
|---|---|
| `factor_graph.py` | the factor graph: factors, uncertainties, Levenberg-Marquardt solver |
| `graph_inputs.py` | inputs of the graph for one trial and kinematic case (camera or IMU) |
| `planar_model.py` | planar segment model, filters, joint moments by inverse dynamics |
| `camera_terms.py` | whole-body kinematic terms from the segments; CoP on the foot line |
| `competitors.py` | competing methods: zero shear (ZS), Newton + proportional split (NP), smooth transition assumption (STA) |
| `pivot_split.py` | competing method: pivot split (PS), after Castellaz et al. |
| `wang_published.py` | competing method: the data set's published pipeline (WP), read from the released files |
| `wang_data.py`, `wang_segments.py` | reading the data set; the camera and IMU segment tables |
| `config.py` | paths |

Steps (in the order of `run_all.py`; outputs in `results/<name>/`):

| Script | Article | Content |
|---|---|---|
| `trials.py` | Sect. 3 | the 53 trials with complete marker data (`results/trials.csv`) |
| `time_shift.py` | Online Resource 1 | residual time shift between insoles and camera kinematics |
| `camera_mc.py` | Table 2 | camera kinematic uncertainties, Monte Carlo |
| `cop_mc.py` | Table 2 | camera CoP placement uncertainty, Monte Carlo |
| `imu_budget.py` | Table 2 | IMU uncertainties from the IMU - camera difference |
| `smoothness_ratio.py` | Table 1 | horizontal-to-vertical smoothness ratio |
| `contact_threshold.py` | Sect. 4.1 | insole contact threshold from kinematic gait events |
| `smoothness_scale.py` | Table 1 | per-trial smoothness uncertainty |
| `run_main.py` | Table 3 | main run: factor graph and competing methods, camera and IMU kinematics |
| `run_ablation.py`, `run_ablation2.py` | Table 4 | ablation |
| `run_sensitivity.py`, `run_sensitivity_smoothness.py` | Table 5 | sensitivity |
| `direction_factor.py` | Sect. 5.5 | where the direction factor acts |
| `planar_vs_lab.py` | Sect. 3 | planar reference against the laboratory inverse dynamics of the data set |
| `hip_center.py` | Sect. 3 | effect of the hip-center rule |
| `timing.py` | Sect. 2.3 | solve times |
| `by_speed.py` | Online Resource 1 | main result by walking speed |
| `figure2_data.py`, `figure2.py` | Fig. 2 | series of Fig. 2 and the figure |
| `paper_tables.py` | Tables 1-5 | LaTeX tables of the article and of Online Resource 1 |
| `paper_numbers.py` | | the numbers of the article, from the result files |
| `verify.py` | | check of every number of the article |

## License

MIT (`LICENSE`). The data set has its own license (above).

## References

Wang H, Basu A, Durandau G, Sartori M (2023) A wearable real-time kinetic measurement sensor setup for human locomotion.
Wearable Technologies 4:e11. <https://doi.org/10.1017/wtc.2023.7>

Wang H, Basu A, Durandau G, Sartori M (2022) Comprehensive kinetic and EMG dataset of daily locomotion with 6 types of
sensors. Zenodo. <https://doi.org/10.5281/zenodo.6457662>
