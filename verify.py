"""Check every number of the article against the result files.

1. Tables: paper_tables.py is rerun in memory and its output must equal the table*.tex files beside the article source.
2. Text: each check pairs a value recomputed by paper_numbers.py (or read from the code) with the value as printed, rounded the
   same way, and the phrase of main.tex in which it is printed (whitespace-insensitive). Facts are checked statements
   (e.g. "PS is the closest competing method in every column").
3. Audit: every number in the body of main.tex (abstract to declarations; comments, equations, citations, labels and URLs
   masked) must lie inside the phrase of some check, literature value or design fact; the rest is listed as uncovered.
The article source (main.tex, main.aux, esm1.tex and the table files) is not part of this repository; give its folder as
the argument. Reads results and code only, writes nothing. Exit code 1 on any FAIL or uncovered number.
Usage: python -I verify.py <folder of the article source>
"""
import inspect
import os
import re
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import config  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import factor_graph as L  # noqa: E402
import paper_numbers as MN  # noqa: E402
import paper_tables as MT  # noqa: E402

# folder of the article source (main.tex, main.aux, esm1.tex, table*.tex), first argument or GAIT_FG_ARTICLE
ARGS = [a for a in sys.argv[1:] if not a.startswith("-")]
MS = os.path.abspath(ARGS[0] if ARGS else os.environ.get("GAIT_FG_ARTICLE", "article"))
TEX = os.path.join(MS, "main.tex")

RESULTS = []      # (kind, name, ok, detail)
PHRASES = []      # (phrase, kind) for the audit


def src_of(path):
    return open(path, encoding="utf-8").read()


def masked(text, patterns):
    """Same-length masking, so positions stay those of the source."""
    for p, flags in patterns:
        text = re.sub(p, lambda m: re.sub(r"[^\n]", " ", m.group(0)), text, flags=flags)
    return text


RAW = src_of(TEX)
NOCOMMENT = masked(RAW, [(r"(?<!\\)%.*$", re.M)])
A0, A1 = RAW.index(r"\begin{abstract}"), RAW.index(r"\bibliographystyle")


def phrase_re(ph):
    parts = [re.escape(w) for w in ph.split()]
    return re.compile(r"\s+".join(parts))


def find(ph):
    return [m.span() for m in phrase_re(ph).finditer(NOCOMMENT, A0, A1)]


def fmt(value, written):
    w = written.lstrip("+")
    nd = len(w.split(".")[1]) if "." in w else 0
    s = f"{value:.{nd}f}"
    return "0" * 0 + (s if s not in ("-0", "-0.0", "-0.00", "-0.000") else s[1:])


def chk(name, value, written, ph):
    """Value, rounded as printed, equals the printed number, and the phrase with it is in main.tex."""
    ok_v = fmt(value, written) == written.lstrip("+")
    ok_p = bool(find(ph)) and written in ph
    RESULTS.append(("value", name, ok_v and ok_p,
                    f"computed {value!r} -> {fmt(value, written)}, printed {written}" + ("" if ok_p else "; PHRASE NOT FOUND")))
    PHRASES.append((ph, "check"))


def fact(name, cond, ph=None):
    ok_p = ph is None or bool(find(ph))
    RESULTS.append(("fact", name, bool(cond) and ok_p, "" if ok_p else "PHRASE NOT FOUND"))
    if ph:
        PHRASES.append((ph, "fact"))


def lit(ph, source):
    """A literature value or a property of the data set, printed as the source gives it; not recomputed."""
    ok = bool(find(ph))
    RESULTS.append(("lit", f"{source}: {ph}", ok, "" if ok else "PHRASE NOT FOUND"))
    PHRASES.append((ph, "lit"))


# ------------------------------------------------------------------------------------------------------------------ tables
def check_tables(N):
    got = {}
    real = MT.write
    MT.write = lambda name, txt: got.__setitem__(name, MT.HEAD + txt)
    try:
        MT.check_values(N["der"])
        MT.table_common(N["der"])
        MT.table_kinematic()
        MT.table_results(N["main"])
        MT.table_ablation(N["abl"])
        MT.table_sensitivity(N["sens"])
        MT.esm_impl(N["der"])
        MT.esm_mc(N["der"])
        MT.esm_contrasts(N["main"])
        MT.esm_speed(N["speed"])
        MT.esm_direction(N["dirn"])
    finally:
        MT.write = real
    esm = masked(src_of(os.path.join(MS, "esm1.tex")), [(r"(?<!\\)%.*$", re.M)])
    for name, txt in got.items():
        disk = src_of(os.path.join(MS, name))
        RESULTS.append(("table", name, disk == txt, "" if disk == txt else "differs from a fresh run of paper_tables.py"))
        doc, where = (esm, "esm1.tex") if name.startswith("esm_") else (NOCOMMENT, "main.tex")
        fact(f"{where} inputs {name}", "\\input{" + name[:-4] + "}" in doc)
    on_disk = {f for f in os.listdir(MS) if re.fullmatch(r"(esm_)?table\d.*\.tex", f)}
    fact("no stale table files on disk", on_disk == set(got), None)
    # the article's numbers that Online Resource 1 quotes (tables and text), against main.aux
    aux = src_of(os.path.join(MS, "main.aux"))
    num = lambda lab: re.search(r"\\newlabel\{" + re.escape(lab) + r"\}\{\{([^}]*)\}", aux).group(1)  # noqa: E731
    for f, frag, lab in (("esm_table3_contrasts.tex", r"Table~3", "tab:results"),
                         ("esm_table5_direction.tex", r"Sect.~5.2", "sec:metrics")):
        want = re.sub(r"[^\d.]", "", frag.split("~")[1])
        fact(f"{f} quotes {frag} = {lab}", frag in got[f] and num(lab) == want)


def check_esm(N):
    """Online Resource 1 (esm1.tex): its text numbers, as main.tex's, and the article numbers it quotes (main.aux)."""
    raw = src_of(os.path.join(MS, "esm1.tex"))
    txt = masked(raw, [(r"(?<!\\)%.*$", re.M)])
    a0, a1 = txt.index(r"\begin{document}"), txt.index(r"\bibliographystyle")
    aux = src_of(os.path.join(MS, "main.aux"))
    num = lambda lab: re.search(r"\\newlabel\{" + re.escape(lab) + r"\}\{\{([^}]*)\}", aux).group(1)  # noqa: E731
    phr = []

    def efind(ph):
        return [m.span() for m in phrase_re(ph).finditer(txt, a0, a1)]

    def ecase(name, cond, ph):
        ok = bool(cond) and bool(efind(ph))
        RESULTS.append(("fact", "esm1: " + name, ok, "" if efind(ph) else "PHRASE NOT FOUND"))
        phr.append(ph)
    import camera_mc as K
    SP = N["speed"]
    ecase("title", True, "Online Resource 1")
    ecase("replicas", K.REPS == 30, r"each error source is simulated on its own in \num{30}")
    ecase("one subject without the 0.75 m/s trial", all(SP[(c, 0.75, k)]["n"] == 8 for c in MN.CASES for k in MN.COLS)
          and num("sec:data") == "3", r"at \SI{0.75}{\metre\per\second} one subject has no trial (Sect.~3)")
    ecase("Sect. 4.2 = Monte Carlo", num("sec:devicevalues") == "4.2", r"on the data's own markers (Sect.~4.2)")
    ecase("Table 2 = kinematic values", num("tab:kinematic") == "2", r"each value of Table~2 is the median")
    ecase("Table 3 = results", num("tab:results") == "3", r"for every error column of Table~3")
    ecase("Sect. 5.5 = ablation", num("sec:res:ablation") == "5.5", r"the removal of the direction factor (Sect.~5.5)")
    mask = masked(txt, [(r"\\(label|ref|cite|input|url)(\[[^\]]*\])?\{[^}]*\}", 0),
                        (r"\$[^$]*\\(dot|sigma|bm|mathbf)[^$]*\$", 0),
                        (r"\\\\\[[\d.]+em\]", 0), (r"\\(left|right)hyphenmin=\d", 0)])  # layout lengths, not numbers
    spans = [sp for ph in phr for sp in efind(ph)]
    unc = []
    for m in re.finditer(r"(?<![\w.\\])[-+]?\d+(?:\.\d+)?", mask[a0:a1]):
        a = a0 + m.start()
        if not any(s0 <= a < s1 for s0, s1 in spans):
            unc.append((raw.count("\n", 0, a) + 1, m.group(0), " ".join(raw[max(0, a - 50):a + 30].split())))
    return unc


# ------------------------------------------------------------------------------------------------------------------- text
def check_text(N):
    M, T, D, AB, SE, TM = N["main"], N["main"]["table"], N["der"], N["abl"], N["sens"], N["time"]
    SP, DN, PL, HP = N["speed"], N["dirn"], N["plab"], N["hip"]
    C = M["contrast"]
    slow_ps = all(SP[(c, v, "stance")]["pct"] > 0 for c in MN.CASES for v in (0.25, 0.5))
    fast_fg = all(SP[(c, v, "stance")]["pct"] < 0 for c in MN.CASES for v in (1.0, 1.25, 1.5))
    pct = lambda case, k: -C[(case, k)]["pct"]  # noqa: E731  (printed as "lower by")
    lw = pd.read_csv(os.path.join(MN.R, "main", "main.csv"))
    speeds = sorted({int(t.split("_")[1]) / 10 / 3.6 for t in lw.trial.unique()})

    # abstract
    chk("solve time, median", TM["seconds_median"], "0.14", r"\SI{0.14}{\second} for a one-minute trial")
    chk("trials", M["n_trials"], "53", r"nine subjects (\num{53} trials) walked on an")
    fact("nine subjects", M["n_subjects"] == 9, r"nine subjects (\num{53} trials)")
    jp = [pct("camera", j) for j in MN.JOINTS]
    chk("camera joint gain, min", min(jp), "12", r"\numrange{12}{31}\,\% below the closest one")
    chk("camera joint gain, max", max(jp), "31", r"\numrange{12}{31}\,\% below the closest one")
    fact("camera: FG lowest in every joint (abstract)", all(C[("camera", j)]["diff"] < 0 for j in MN.JOINTS),
         "graph has lower joint-moment errors than every competing method at the ankle, knee and hip")
    for j, w in (("knee", "11"), ("hip", "20")):
        chk(f"imu {j} gain", pct("imu", j), w, r"it is lowest at the knee and hip, \num{11} and \num{20}\,\% below")
    fact("imu ankle: WP best, better than FG", C[("imu", "ankle")]["best"] == "wang" and C[("imu", "ankle")]["diff"] > 0,
         "WP is better")
    knee0 = C[("imu", "knee")]["lo"] < 0 < C[("imu", "knee")]["hi"]
    hip_only = knee0 and C[("imu", "hip")]["hi"] < 0      # with IMU kinematics only the hip interval excludes zero
    fact("discussion: WP better at the ankle with IMU, only the hip gain clear",
         C[("imu", "ankle")]["best"] == "wang" and C[("imu", "ankle")]["diff"] > 0 and hip_only,
         "although there only the hip gain is clear and WP is\nbetter at the ankle")
    fact("abstract: imu only the hip gain clear", hip_only, "below, though only the hip gain is clear")
    fact("conclusions: imu only the hip gain clear", hip_only, "though there only the hip\ngain is clear")
    rz = M[("camera", "fz_graph")] / M[("camera", "fz_insole")]
    fact("abstract: camera Fz error halved (ratio 0.45-0.55)", 0.45 < rz < 0.55,
         "It also halves the insole's vertical-force error with camera kinematics")

    # model and solution
    fact("seven factors, two scales and an offset per trial", True, "seven factors, with two scales and an offset per trial")
    lit("(Intel Core i7-13700H)", "timing machine, read from the system")
    spsrc = src_of(os.path.join(HERE, "planar_model.py"))
    chk("joint moments: force at the foot CoM at or below (N)", float(re.search(r"^F_CONTACT = ([\d.]+)", spsrc, re.M).group(1)),
        "20", r"(a force of \SI{20}{\newton} or less, whose CoP is unreliable, at the foot's CoM)")
    fact("joint moments: CoP replaced by the foot CoM out of contact", 'np.where(on, copx, foot["com"][:, 0])' in spsrc)
    fact("design: values fixed before scoring", True, "all were fixed before the scores were computed.")
    fact("sigma_sm varied: halved and doubled", "smoothness" in MN.SENS and MT.ends()[("camera", "SM_SCALE", "low")] == 0.5
         and MT.ends()[("camera", "SM_SCALE", "high")] == 2.0)
    fact("design: pivot follows the CoM", 'kin["xc"][D] + p["PIVOT_FWD"]' in src_of(os.path.join(HERE, "factor_graph.py")),
         "moves the pivot back with the CoM")
    fact("design: identity of the summed direction residuals (direction_factor.py)", "(Hdot - b - d_x S_z) / z_G"
         in src_of(os.path.join(HERE, "direction_factor.py")),
         r"so through the moment balance the factor also asks $\dot{H}_k - b$ to be close to $d_x S_{z,k}$ in double support")
    fact("6 Hz fourth-order Butterworth", "sp_.lowpass(v, 6.0, FS)" in src_of(os.path.join(HERE, "camera_terms.py"))
         and "def lowpass(x, fc, fs, order=4):" in spsrc,
         r"zero-phase Butterworth filter at \SI{6}{\hertz}, the cut-off customary in gait analysis since Winter et al.")
    lit(r"the cut-off customary in gait analysis since Winter et al.\ \cite{winter1974}, and differentiated", "Winter 1974, 6 Hz cut-off")
    fsrc = src_of(os.path.join(HERE, "factor_graph.py"))
    fact("start: scales 1, offset 0 (zeros), Sx in proportion to measured Fz",
         'v[self.cols["gR"]] = v[self.cols["gL"]] = 1.0' in fsrc and "v = np.zeros(self.N)" in fsrc
         and '(self.kin["Sx"] * fz[s] / tot)' in fsrc, "scales of 1 and a zero offset")
    fact("SciPy sparse LU", "from scipy.sparse.linalg import splu" in fsrc, "implemented in Python with SciPy's sparse LU factorization")
    fact("tolerance 1e-10", L.TOL == 1e-10, r"less than $10^{-10}$ of itself")
    chk("frames, median", TM["frames_median"], "5954", r"\num{5954} frames and \num{23724} unknowns (medians)")
    chk("unknowns, median", TM["unknowns_median"], "23724", r"\num{5954} frames and \num{23724} unknowns (medians)")
    chk("solve time, max", TM["seconds_max"], "0.23", r"solves in \SI{0.14}{\second} (median; at most \SI{0.23}{\second})")
    chk("solve time, median (2)", TM["seconds_median"], "0.14", r"solves in \SI{0.14}{\second} (median; at most")
    fact("every solve converged", M["converged"] == M["n_solves"], "every solve converged")
    chk("iterations, median", M["it_median"], "6", r"in \num{6} iterations (median; at most \num{11})")
    chk("iterations, max", M["it_max"], "11", r"in \num{6} iterations (median; at most \num{11})")
    for ph in ("They are Newton's and Euler's laws for the whole body", "It expresses the regularity of double support",
               "The measured vertical force is the true one times the scale", "It asks the forces to change smoothly",
               "(dir in double support only)", "The balances also link every frame to the kinematics",
               "the sensor factors to the", "at $z = 0$", "both CoPs lie at $z = 0$",
               "at the height of the CoM ($d_z = 0$)", "three consecutive frames"):
        fact("design: " + ph, True, ph)

    # values
    chk("pivot height (cm)", 100 * L.MODEL["PIVOT_UP"], "0.0", r"$0.0 \pm 7.2$\,cm above")
    chk("pivot ahead (cm)", 100 * L.MODEL["PIVOT_FWD"], "5.1", r"$5.1 \pm 1.7$\,cm ahead of the CoM")
    lit(r"$0.0 \pm 7.2$\,cm above and $5.1 \pm 1.7$\,cm ahead of the CoM", "Vielemeyer 2021 (spread)")
    fact("rho in factor_graph = 0.55", L.MODEL["FX_RATIO"] == 0.55)
    ph = r"(interquartile range \numrange{0.46}{0.70})"
    chk("smoothness ratio, lower quartile", D["ratio_q1"], "0.46", ph)
    chk("smoothness ratio, upper quartile", D["ratio_q3"], "0.70", ph)
    e = MT.ends()
    vals = {k: L.MODEL[k] for k in ("PIVOT_FWD", "SDIR", "FX_RATIO")}
    half_double = all(np.isclose(e[(k, "low")], v / 2) and np.isclose(e[(k, "high")], 2 * v) for k, v in vals.items())
    for case, dev in (("camera", L.CAMERA), ("imu", L.IMU)):
        half_double &= all(np.isclose(e[(case, k, "low")], dev[k] / 2) and np.isclose(e[(case, k, "high")], 2 * dev[k])
                           for k in ("SX", "SZ", "SH", "D_STD", "SCOP", "SI", "G_STD", "F_CONTACT"))
    fact("bounds: half and double; pivot height +-7.2 cm (source SD)",
         half_double and L.MODEL["PIVOT_UP"] == 0 and e[("PIVOT_UP", "low")] == -0.072 and e[("PIVOT_UP", "high")] == 0.072,
         "The bounds are half and double each value; the pivot height, which is zero, moves by the source's standard "
         "deviation instead")
    fact("threshold in factor_graph = best-agreement threshold", L.INSOLE["F_CONTACT"] == D["thr_chosen"])
    chk("timing error at threshold (ms)", D["thr_rms_ms"], "27.8", r"with a timing error of \SI{27.8}{\milli\second}")
    fact("insole Fz sigma = midpoint of 10-11 %BW", np.isclose(L.INSOLE["SI"], (10 + 11) / 2 / 100),
         r"\numrange{10}{11}\,\BW\ for insoles scaled to body weight without force-plate calibration")
    fact("scale prior = midpoint of 1.1-4.4 %, rounded", round((1.1 + 4.4) / 2 / 100, 2) == L.INSOLE["G_STD"],
         r"the \numrange{1.1}{4.4}\,\% mean biases of the walking peak, valley and impulse forces")
    chk("scale prior", L.INSOLE["G_STD"], "0.03", "rounded to 0.03")
    lit(r"the sensor's error, \SI{15}{\milli\metre} in walking with the insole located from another trial \cite{morin2022}", "Morin 2022")
    lit(r"half a body height above the pelvis \cite{castellaz2024}", "Castellaz 2024, pivot 0.5 BH")
    lit(r"without their correction force, which needs force-plate calibration trials", "Forner-Cordero 2006, F_corr")
    fact("15 mm sensor CoP error used in the camera CoP total", "0.015 ** 2" in src_of(os.path.join(HERE, "paper_numbers.py")))
    chk("camera CoP placement (mm)", D["cop_place_cam_mm"], "8.5",
        r"the placement error, \SI{8.5}{\milli\metre}, comes from the same Monte Carlo simulation")
    fact("time-shift correlation threshold 0.5 (Online Resource 1)", "r_peak >= 0.5" in src_of(os.path.join(HERE, "paper_numbers.py")))
    fact("CoP placement perturbs markers, shoe position and time shift", True,
         "the insole's position in the shoe and the insole--marker time shift perturbed")
    chk("IMU CoP placement (mm)", D["imu_cop_place_mm"], "43.7", r"with IMU kinematics it is \SI{43.7}{\milli\metre}, from the")
    import camera_mc as K0
    chk("Monte Carlo replicas", K0.REPS, "30", r"in \num{30} replicas per trial that recompute")
    fact("CoP Monte Carlo uses the same replicas", "K.REPS" in src_of(os.path.join(HERE, "cop_mc.py")))
    import camera_mc as KM
    step = (10.7 + 15.1) / 2      # Moticon OpenGo size steps 10.7-15.1 mm (results/sources/mc_input_sources.json, shoe_sync)
    fact("shoe displacement SD = SD of a uniform within half a size step", np.isclose(round(step / 2 / np.sqrt(3), 1),
                                                                                   KM.LEVELS["central"]["cop"]["shoe_mm"]))


    # data and cases
    fact("six speeds, 0.25 to 1.5 m/s in steps of 0.25", len(speeds) == 6 and np.allclose(np.diff(speeds), 0.25)
         and np.isclose(speeds[0], 0.25) and np.isclose(speeds[-1], 1.5),
         r"at six speeds, \SIrange{0.25}{1.5}{\metre\per\second} in steps of \SI{0.25}{\metre\per\second}")
    lit("pressure insoles (Moticon), which give", "Wang 2023, device")
    lit(r"released synchronized at \SI{100}{\hertz}", "Wang 2023, data")
    chk("trials recorded (9 x 6)", M["n_subjects"] * len(speeds), "54", r"One of the \num{54} trials lacks valid")
    chk("trials scored", M["n_trials"], "53", r"the remaining \num{53} last about one minute each")
    fact("last second not scored", "info[\"win\"][n - int(R.FS):] = False" in src_of(os.path.join(HERE, "graph_inputs.py")),
         "the first and last second of each are not scored")
    chk("double-support share, median", 100 * M["ds_share"], "32", r"double support in \num{32}\,\% of the frames")
    fact("hip centre 0.048 H below the pelvis mean", "0.048 * height" in src_of(os.path.join(HERE, "wang_segments.py")),
         r"\num{0.048} body heights below the mean of the pelvis markers")
    ph = r"about \SI{4}{\centi\metre} behind that regression's and the whole-body CoM about \SI{3}{\centi\metre} behind"
    chk("hip: shift to Harrington's AP position (cm)", HP["shift_mm"] / 10, "4", ph)
    chk("hip: CoM shift (cm)", HP["dxG_mm"] / 10, "3", ph)
    chk("hip: constant moment-balance error (m BW)", -HP["moment_mBW"], "0.03",
        r"adds a constant error of about \num{0.03}\,m\,BW to the")
    ph = r"by \num{0.119}, \num{0.114} and \num{0.201}\,\Nmkg\ at the ankle, knee and hip (root-mean-square error, RMSE)"
    for j, w in (("ankle", "0.119"), ("knee", "0.114"), ("hip", "0.201")):
        chk(f"planar vs laboratory, {j}", PL[(j, "planar_lab")], w, ph)
    near = max(PL[(j, "wp_planar")] - PL[(j, "wp_lab")] for j in MN.JOINTS)
    fact("WP at no joint nearer the laboratory solution than ours by more than 0.01", near < 0.01,
         r"is at no joint nearer to the laboratory solution than to ours by more than \num{0.01}\,\Nmkg")
    import imu_budget as W
    fact("IMU alignment threshold 0.5 (Online Resource 1)", W.R_MIN == 0.5)
    chk("IMU trials with own time shift", D["imu_reliable"], "38", r"\num{38} trials use their own shift")

    # competitors and metrics
    import pivot_split as CP
    import run_main as RW
    sig = inspect.signature(CP.total_force).parameters
    cps = inspect.signature(CP.castellaz).parameters
    fact("PS smoothed before (moving means) and after (Savitzky-Golay) the split, original 2025 settings",
         sig["ap_s"].default == 0.1 and sig["v_s"].default == 0.05 and cps["how"].default == sig["how"].default == "2025"
         and cps["sg"].default == 15, "The forces are smoothed before and after the split with the original's filters")
    fact("PS pivot half a body height above the pelvis", CP.VPP_BH == 0.5, "half a body height above the pelvis")
    ph = (r"with the \SI{0.1}{\second} horizontal window of its overground tasks rather than the \SI{0.3}{\second} it used "
          r"against belt vibration on the treadmill")
    chk("PS horizontal window (s)", sig["ap_s"].default, "0.1", ph)
    lit(ph, "Castellaz 2025, 0.3 s window on its treadmill trials")
    fact("Online Resource 1 exists and is compiled", os.path.exists(os.path.join(MS, "esm1.tex")), "Online Resource 1")
    fact("plate contact 20 N", RW.PLATE_CONTACT == 20.0, r"vertical force exceeds \SI{20}{\newton}")
    fact("CoP scored where the right belt carries > 10 % BW", 'ref["R"]["Fy"] > 0.1 * BW' in src_of(os.path.join(HERE, "run_main.py")),
         r"the right belt carries more than \num{10}\,\% of body weight")
    fact("95 % t intervals", "stats.t.ppf(0.975" in src_of(os.path.join(HERE, "paper_numbers.py")),
         r"\num{95}\,\% $t$ intervals over the nine subjects")

    # camera results
    fact("camera: FG lowest in every column", all(C[("camera", k)]["diff"] < 0 for k in MN.COLS)
         and T[("camera", "graph", "fz_stance")] < T[("camera", "pivot", "fz_stance")],
         "has the lowest error of all methods in every column")
    fact("camera: PS closest in every column", all(C[("camera", k)]["best"] == "pivot" for k in MN.COLS),
         "PS is the closest competing method in every column")
    ph = (r"against PS the ankle, knee and hip errors are lower by \num{12}, \num{31} and \num{27}\,\%, and all three "
          r"intervals exclude zero")
    for j, w in (("ankle", "12"), ("knee", "31"), ("hip", "27")):
        chk(f"camera {j} gain", pct("camera", j), w, ph)
    fact("camera: all three joint intervals exclude zero", all(C[("camera", j)]["hi"] < 0 for j in MN.JOINTS), ph)
    ph = r"In shear the gain is clear only in single support, \num{20}\,\%"
    chk("camera ss gain", pct("camera", "ss"), "20", ph)
    fact("camera ss interval excludes zero", C[("camera", "ss")]["hi"] < 0, ph)
    ph = r"over stance and in double support, \num{8} and \num{5}\,\%, the intervals include zero"
    for k, w in (("stance", "8"), ("ds", "5")):
        chk(f"camera {k} gain", pct("camera", k), w, ph)
        fact(f"camera {k}: interval includes zero", C[("camera", k)]["lo"] < 0 < C[("camera", k)]["hi"], ph)
    fact("speed: PS lower over stance at 0.25 and 0.5 m/s, FG from 1.0 m/s on, both kinematics", slow_ps and fast_fg,
         r"over stance PS has the lower error at \num{0.25} and \SI{0.5}{\metre\per\second}, the factor graph from "
         r"\SI{1.0}{\metre\per\second} on, with either kinematics")
    ok = all(SP[("camera", v, j)]["pct"] < 0 for v in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5) for j in MN.JOINTS
             if (v, j) != (0.25, "hip")) and abs(SP[("camera", 0.25, "hip")]["pct"]) < 2
    fact("camera: joint gains at every speed, hip at 0.25 m/s level (|diff| < 2 %)", ok,
         r"The joint-moment gains hold at every speed, except at the hip at \SI{0.25}{\metre\per\second}, where the two "
         r"methods are level")
    fact("speeds of the split are the trial speeds", sorted({v for (_, v, _) in SP}) == [round(x, 2) for x in speeds])
    ph = (r"over stance is \num{6.02}\,\BW, against \num{11.82} for the insole as measured and \num{9.34} for PS's "
          r"estimate")
    chk("camera FG Fz", M[("camera", "fz_graph")], "6.02", ph)
    chk("camera insole Fz", M[("camera", "fz_insole")], "11.82", ph)
    chk("camera PS Fz", T[("camera", "pivot", "fz_stance")], "9.34", ph)
    ph = r"placed, \SI{27.0}{\milli\metre} against \SI{28.3}{\milli\metre}, in all nine subjects"
    chk("camera FG CoP (mm)", M[("camera", "cop_graph")], "27.0", ph)
    chk("camera insole CoP (mm)", M[("camera", "cop_insole")], "28.3", ph)
    fact("camera CoP lower in all nine subjects", C[("camera", "cop")]["n_better"] == 9 == C[("camera", "cop")]["n"], ph)
    fact("camera FG CoP slightly lower than insole CoP", 0 < M[("camera", "cop_insole")] - M[("camera", "cop_graph")] < 3,
         "Its CoP error is slightly lower than that of the insole's CoP as")
    cop = {}
    for case in MN.CASES:
        S = MN.subj(lw, case)
        dcop = S.graph_cop_stance_mm - S.insole_cop_stance_mm
        cop[case] = dict(g=float(S.graph_cop_stance_mm.mean()), i=float(S.insole_cop_stance_mm.mean()),
                         ci=MN.t95(dcop), nb=int((dcop < 0).sum()), n=int(np.isfinite(dcop).sum()))
    fact("camera CoP mean from subject means equals M", np.isclose(cop["camera"]["g"], M[("camera", "cop_graph")]))
    fact("CoP contrast (Online Resource 1) equals the direct one", all(np.isclose(cop[c]["ci"][1], C[(c, "cop")]["lo"])
                                                                         and cop[c]["nb"] == C[(c, "cop")]["n_better"]
                                                                         for c in MN.CASES))
    ph = r"The estimated scales, \num{0.963} and \num{0.956} (right and left, medians over trials)"
    chk("camera gR, median", M[("camera", "gR")], "0.963", ph)
    chk("camera gL, median", M[("camera", "gL")], "0.956", ph)
    chk("insoles read low by (1 - mean scale), %", 100 * (1 - (M[("camera", "gR")] + M[("camera", "gL")]) / 2), "4",
        r"mean that the insoles read about 4\,\% low")
    fact("offset of the main run equals that of the direction study", np.isclose(M[("camera", "delta")],
                                                                                    DN[("camera", "main", "b")]))
    import figure2_data as MF
    subj, trial, _ = MF.pick_trial()
    ftxt = src_of(os.path.join(config.RESULTS, "figure2", "fig2_trial.txt"))
    fact("Fig. 2 trial is the one the rule picks", f"trial {subj} {trial};" in ftxt)
    chk("Fig. 2 speed (m/s)", MF.wa_speed(trial), "1.0", r"One trial at \SI{1.0}{\metre\per\second}")

    # IMU results
    fact("imu: FG lowest in shear, knee, hip", all(C[("imu", k)]["diff"] < 0 for k in ("stance", "ds", "ss", "knee", "hip")),
         "again has the lowest shear error and the lowest knee and hip errors")
    fact("imu: PS closest in shear, knee, hip", all(C[("imu", k)]["best"] == "pivot"
                                                     for k in ("stance", "ds", "ss", "knee", "hip")))
    ph = (r"Against PS the gain is clear at the hip, \num{20}\,\%; at the knee, \num{11}\,\%, and in shear, "
          r"\numrange{8}{15}\,\%, the intervals include zero")
    chk("imu hip gain (2)", pct("imu", "hip"), "20", ph)
    chk("imu knee gain (2)", pct("imu", "knee"), "11", ph)
    chk("imu shear gain, min", min(pct("imu", k) for k in MN.SHEAR), "8", ph)
    chk("imu shear gain, max", max(pct("imu", k) for k in MN.SHEAR), "15", ph)
    fact("imu: knee and shear intervals include zero", all(C[("imu", k)]["lo"] < 0 < C[("imu", k)]["hi"]
                                                           for k in MN.SHEAR + ("knee",)), ph)
    fact("imu knee: FG lower in only 4 of 9 subjects below 0.75 m/s",
         all(SP[("imu", v, "knee")]["n_better"] == 4 and SP[("imu", v, "knee")]["n"] == 9 for v in (0.25, 0.5)),
         r"below \SI{0.75}{\metre\per\second} the factor graph has the lower knee error in only four of the nine subjects")
    fact("imu: hip interval excludes zero, knee interval includes zero",
         C[("imu", "hip")]["hi"] < 0 and C[("imu", "knee")]["lo"] < 0 < C[("imu", "knee")]["hi"],
         "Against PS the gain is clear at the hip")
    fact("imu ankle: FG below ZS, NP, STA and PS", T[("imu", "graph", "ankle")]
         < min(T[("imu", m, "ankle")] for m in ("zero", "newton", "ren", "pivot")),
         "At the ankle its error is lower than those of ZS, NP, STA and PS")
    chk("imu WP ankle", T[("imu", "wang", "ankle")], "0.108", r"\num{0.108} against \num{0.203}\,\Nmkg")
    chk("imu FG ankle", T[("imu", "graph", "ankle")], "0.203", r"\num{0.108} against \num{0.203}\,\Nmkg")
    chk("imu ZS ankle", T[("imu", "zero", "ankle")], "0.249",
        r"ZS, whose ankle error with our IMU kinematics is \num{0.249}\,\Nmkg")
    ph = r"that of the insole's CoP as placed, \SI{104.1}{\milli\metre} against \SI{101.7}{\milli\metre}"
    chk("imu FG CoP (mm)", cop["imu"]["g"], "104.1", ph)
    chk("imu insole CoP (mm)", cop["imu"]["i"], "101.7", ph)
    fact("imu FG CoP not lower than insole CoP", cop["imu"]["g"] >= cop["imu"]["i"],
         "its CoP error is not lower than that of the insole's CoP as placed")
    fact("WP worse than FG and PS at knee and hip", all(T[("imu", "wang", j)] > T[("imu", m, j)] for j in ("knee", "hip")
                                                        for m in ("graph", "pivot")),
         "At the knee and hip WP is worse than both the factor graph and PS")
    ph = r"factor graph is \num{7.32}\,\BW, against \num{11.82} for the insole and \num{12.66} for PS"
    chk("imu FG Fz", M[("imu", "fz_graph")], "7.32", ph)
    chk("imu insole Fz", M[("imu", "fz_insole")], "11.82", ph)
    chk("imu PS Fz", T[("imu", "pivot", "fz_stance")], "12.66", ph)

    # ablation
    single = MN.ABL[:4]
    fact("each removal raises the mean moment error in both cases; the ankle error falls without the moment balance",
         all(AB[(c, a)]["moments"] > 0 for c in MN.CASES for a in single)
         and all(AB[(c, "no moment balance")]["abs"]["ankle"] < T[(c, "graph", "ankle")] for c in MN.CASES),
         "Each removal raises the mean joint-moment error in both kinematic cases, though not every joint error; the ankle "
         "error, for one, falls without the moment balance")
    fact("no direction: single-support shear error falls, both cases",
         all(AB[(c, "no direction")]["abs"]["ss"] < T[(c, "graph", "ss")] for c in MN.CASES),
         "Although it acts in double support only, it raises the single-support shear error too")
    rel = lambda c, k: DN[(c, "no direction")][k] / DN[(c, "main")][k] - 1  # noqa: E731
    fact("no direction: vertical force and CoP change by less than 5 %",
         all(abs(rel(c, k)) < 0.05 for c in MN.CASES for k in ("fz_stance", "cop_stance_mm")),
         "Without it the vertical force and the CoP hardly change")
    fact("no direction: knee and hip errors rise more in single than in double support",
         all(rel(c, f"{j}_ss") > max(rel(c, f"{j}_ds"), 0) for c in MN.CASES for j in ("knee", "hip")),
         "the knee and hip errors rise mostly in single support")
    fact("direction study reproduces the ablation", all(np.isclose(DN[(c, "no direction")]["shear_stance"],
                                                                   AB[(c, "no direction")]["abs"]["stance"]) for c in MN.CASES))
    ph = (r"which the kinematics give as \num{-0.033}\,m\,BW with camera and \num{-0.031} with IMU kinematics "
          r"(medians over trials; negative in \num{52} and \num{53} of the \num{53} trials)")
    chk("gap, camera (m BW)", DN[("camera", "gap")], "-0.033", ph)
    chk("gap, imu (m BW)", DN[("imu", "gap")], "-0.031", ph)
    chk("gap negative, camera", DN[("camera", "gap_neg")], "52", ph)
    chk("gap negative, imu", DN[("imu", "gap_neg")], "53", ph)
    chk("gap trials", DN[("camera", "n")], "53", ph)
    ph = (r"the offset, also a median, is \num{-0.020}\,m\,BW with camera and \num{-0.009} with IMU kinematics "
          r"with the factor, and \num{-0.009} and \num{0.028} without it")
    for c, arm, w in (("camera", "main", "-0.020"), ("imu", "main", "-0.009"), ("camera", "no direction", "-0.009"),
                      ("imu", "no direction", "0.028")):
        chk(f"offset b, {c}, {arm}", DN[(c, arm, "b")], w, ph)
    fact("the offset moves toward the gap with the factor",
         all(abs(DN[(c, "main", "b")] - DN[(c, "gap")]) < abs(DN[(c, "no direction", "b")] - DN[(c, "gap")])
             for c in MN.CASES), "It also pulls the moment offset toward")
    fact("moment balance: shear and moments worse in all nine, both cases",
         all(AB[(c, "no moment balance")]["n_shear"] == 9 and AB[(c, "no moment balance")]["n_moments"] == 9
             for c in MN.CASES), "grow in all nine subjects, with either kinematics")
    sm = [AB[(c, "no force smoothness")][q] for c in MN.CASES for q in ("shear", "moments")]
    fact("smoothness helps everywhere, least in IMU moments", min(sm) > 0 and min(sm) == AB[("imu",
         "no force smoothness")]["moments"], "Smoothness helps everywhere, least in the moments with IMU kinematics")
    chk("camera, no scale and offset: moments", AB[("camera", "no scale and offset")]["moments"], "21.8",
        r"\num{21.8}\,\% higher without them")
    fact("no scale and offset: hardly with IMU (< 2 %)", abs(AB[("imu", "no scale and offset")]["moments"]) < 2,
         "but hardly with IMU kinematics")
    fact("scales and offset cost a little in shear (removing them lowers it by less than 10 %)",
         all(-10 < AB[(c, "no scale and offset")]["shear"] < 0 for c in MN.CASES), "and they cost a little in shear")
    ph = (r"lowers the stance shear error by \num{23.2}\,\% with camera and \num{16.5}\,\% with IMU kinematics, but raises the "
          r"mean moment error by \num{8.6} and \num{22.2}\,\%")
    chk("camera, no direction: shear", -AB[("camera", "no direction")]["shear"], "23.2", ph)
    chk("imu, no direction: shear", -AB[("imu", "no direction")]["shear"], "16.5", ph)
    chk("camera, no direction: moments", AB[("camera", "no direction")]["moments"], "8.6", ph)
    chk("imu, no direction: moments", AB[("imu", "no direction")]["moments"], "22.2", ph)
    allr = "balances and sensors only"
    fact("all four removed: shear exactly NP", all(np.isclose(AB[(c, allr)]["abs"][k], T[(c, "newton", k)], rtol=0,
                                                              atol=1e-9) for c in MN.CASES for k in MN.SHEAR),
         "the shear error is exactly that of NP")
    fact("all four removed: knee and hip moments below NP", all(AB[(c, allr)]["abs"][j] < T[(c, "newton", j)]
                                                                for c in MN.CASES for j in ("knee", "hip")),
         "the knee and hip moments remain somewhat better than NP's")

    # sensitivity
    ends = [(r, e) for r in MN.SENS for e in ("low", "high")]
    fact("eighteen ends", len(ends) == 18, "at seven of the eighteen ends")
    fact("eighteen ends (method)", len(ends) == 18, "eighteen ends in all")
    fact("two groups in the sensitivity study", sum(x in ("kinematic uncertainties", "insole Fz and scale") for x in MN.SENS) == 2,
         "one of the two groups given in")
    ph = (r"up to \num{23}\,\% with camera and \num{32}\,\% with IMU kinematics, and the mean joint-moment error by up to "
          r"\num{12} and \num{28}\,\%")
    for c, ws, wm in (("camera", "23", "12"), ("imu", "32", "28")):
        chk(f"{c} sensitivity, max shear change", max(abs(SE[(c, r, e)]["shear"]) for r, e in ends), ws, ph)
        chk(f"{c} sensitivity, max moment change", max(abs(SE[(c, r, e)]["moments"]) for r, e in ends), wm, ph)
    jl = {c: [(r, e) for r, e in ends if set(SE[(c, r, e)]["lost"]) & set(MN.JOINTS)] for c in MN.CASES}
    jn = {c: {j for r, e in ends for j in SE[(c, r, e)]["lost"] if j in MN.JOINTS} for c in MN.CASES}
    fact("camera: moment lead lost only at the ankle, at kinematic-low and CoP-high",
         jn["camera"] == {"ankle"} and set(jl["camera"]) == {("kinematic uncertainties", "low"), ("CoP", "high")},
         "it loses them only at the ankle, at two ends, with the balance and offset uncertainties halved or the CoP "
         "uncertainty doubled")
    fact("imu: moment lead lost only at the knee, at three ends", jn["imu"] == {"knee"} and len(jl["imu"]) == 3,
         "with IMU kinematics only at the knee, at three ends")
    sl = {c: [(r, e) for r, e in ends if set(SE[(c, r, e)]["lost"]) & set(MN.SHEAR)] for c in MN.CASES}
    fact("shear lead lost at 7 (camera) and 4 (imu) ends", len(sl["camera"]) == 7 and len(sl["imu"]) == 4,
         "some shear column is lost at seven of the eighteen ends with camera and at four with IMU kinematics")
    main3 = {("pivot forward", "high"), ("direction deviation", "low"), ("kinematic uncertainties", "high")}
    fact("shear lost at pivot-forward high, direction low, kinematic high in both cases",
         all(main3 <= set(sl[c]) for c in MN.CASES), "mainly when the pivot is moved forward, the direction uncertainty "
         "halved or the balance and offset uncertainties doubled")
    fact("ends: low halves, high doubles (direction, pivot forward, kinematic, CoP)",
         all(s in src_of(os.path.join(HERE, "run_sensitivity.py")) for s in
             ('PIVOT_FWD=m["PIVOT_FWD"] * 2', 'SDIR=m["SDIR"] / 2', '(("low", 0.5), ("high", 2.0))')))
    chk("insole row, max change", max(max(abs(SE[(c, "insole Fz and scale", e)]["shear"]),
                                          abs(SE[(c, "insole Fz and scale", e)]["moments"]))
                                      for c in MN.CASES for e in ("low", "high")), "1.9",
        r"change neither the shear nor the mean moment error by more than \num{1.9}\,\%")

    # discussion and conclusions
    r = M[("camera", "fz_graph")] / M[("camera", "fz_insole")]
    fact("camera Fz error halved (ratio 0.45-0.55)", 0.45 < r < 0.55, "whose error is halved with camera kinematics")
    fact("conclusions: halves", 0.45 < r < 0.55, "it halves the insole's vertical-force error with camera kinematics")
    only_ss = C[("camera", "ss")]["hi"] < 0 and all(C[(c, k)]["lo"] < 0 < C[(c, k)]["hi"] for c, k in
                                                    (("camera", "stance"), ("camera", "ds"), ("imu", "stance"),
                                                     ("imu", "ds"), ("imu", "ss")))
    fact("shear: separated from PS only in single support with camera kinematics", only_ss,
         "In shear the data separate the factor graph from the pivot split only in single support with camera kinematics")
    fact("conclusions: shear separated from PS only in single support with camera kinematics", only_ss,
         "In shear the data separate it from the pivot split only in single support with camera kinematics")
    fact("discussion: the pivot split is better at the slowest speeds", slow_ps,
         "and at the slowest speeds the pivot split is better")
    fact("discussion: the direction factor does not act through Fz or CoP",
         all(abs(DN[(c, "no direction")][k] / DN[(c, "main")][k] - 1) < 0.05 for c in MN.CASES
             for k in ("fz_stance", "cop_stance_mm")),
         "since the vertical force and the CoP hardly change without it")
    fact("limitations: the pivot's speed above every speed here", max(speeds) < 1.7, "faster than every speed here")
    lit(r"overground walking at \SI{1.7}{\metre\per\second}", "Vielemeyer 2021")
    lit("(reference 2021.57)", "Wang 2023, ethics")


# ------------------------------------------------------------------------------------------------------------------ audit
def audit():
    mask = masked(NOCOMMENT, [
        (r"\\begin\{(equation|align)\}.*?\\end\{(equation|align)\}", re.S),
        (r"\\(label|ref|eqref|cite|input|includegraphics|url|resizebox)(\[[^\]]*\])?\{[^}]*\}(\{[^}]*\})?", 0),
        (r"\\begin\{figure\}\[[^\]]*\]", 0),
        (r"\$[^$]*\\(theta|sigma|Delta|mathcal|tilde|mathbf|bm|dot)[^$]*\$", 0),   # symbols with sub/superscripts
    ])
    spans = [s for ph, _ in PHRASES for s in find(ph)]
    out = []
    for m in re.finditer(r"(?<![\w.\\])[-+]?\d+(?:\.\d+)?", mask[A0:A1]):
        a = A0 + m.start()
        if any(s0 <= a < s1 for s0, s1 in spans):
            continue
        line = RAW.count("\n", 0, a) + 1
        ctx = " ".join(RAW[max(0, a - 50):a + 30].split())
        out.append((line, m.group(0), ctx))
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    N = MN.all_numbers()
    check_tables(N)
    check_text(N)
    unc_esm = check_esm(N)
    bad = [r for r in RESULTS if not r[2]]
    for kind, name, ok, det in RESULTS:
        if not ok or "-v" in sys.argv:
            print(f"{'ok  ' if ok else 'FAIL'} [{kind}] {name}  {det}")
    unc = audit()
    print(f"\n{len(RESULTS)} checks: {len(RESULTS) - len(bad)} ok, {len(bad)} FAIL "
          f"({sum(r[0] == 'value' for r in RESULTS)} values, {sum(r[0] == 'fact' for r in RESULTS)} facts, "
          f"{sum(r[0] == 'table' for r in RESULTS)} tables, {sum(r[0] == 'lit' for r in RESULTS)} literature/data)")
    print(f"audit: {len(unc)} numbers in main.tex outside every checked phrase")
    for line, num, ctx in unc:
        print(f"  line {line:4d}  {num:>8s}   {ctx}")
    print(f"audit: {len(unc_esm)} numbers in the text of esm1.tex outside every checked phrase (its tables are checked above)")
    for line, num, ctx in unc_esm:
        print(f"  esm1 line {line:4d}  {num:>8s}   {ctx}")
    bad = [r for r in RESULTS if not r[2]]
    sys.exit(1 if bad or unc or unc_esm else 0)


if __name__ == "__main__":
    main()
