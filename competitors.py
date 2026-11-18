"""Competitors NP and STA (Sect. 5.1): each estimates the horizontal force of each foot from the whole-body horizontal
force S; the vertical force and the CoP are the insole's.

  newton_prop   NP: S split in proportion to the measured vertical force (planar_model.newton_baseline).
  sta_split     STA: the smooth transition assumption of Ren et al. (2008) in double support; formula and source in its
                docstring.
ZS (zero shear) needs no function; PS is in pivot_split.py and WP in wang_published.py.
"""
import numpy as np

import planar_model as sp_


def runs(mask):
    """Start and end (exclusive) indices of the True runs of a boolean array."""
    m = np.asarray(mask, bool).astype(np.int8)
    d = np.diff(np.concatenate([[0], m, [0]]))
    return np.where(d == 1)[0], np.where(d == -1)[0]


def contact(FzR, FzL, thr=None):
    """Foot-on-ground masks from the vertical force."""
    thr = sp_.F_CONTACT if thr is None else thr
    return np.asarray(FzR) > thr, np.asarray(FzL) > thr


# ----------------------------------------------------------------------------- NP


def newton_prop(S, FzR, FzL):
    """NP: S split by vertical load (planar_model.newton_baseline, unchanged)."""
    return sp_.newton_baseline(S, FzR, FzL)


# ----------------------------------------------------------------------------- STA


K1 = np.exp(4.0 / 9.0)                    # Ren et al. 2008 AP constants (see sta_ap_shape)
K2 = 0.5 * K1 * np.exp(-16.0 / 9.0)


def sta_ap_shape(tau):
    """Ren et al. (2008) anteroposterior transition function, written on normalized DS time.

    Ren's form (absolute time t from contralateral heel strike, T_ds = HALF the double-support
    duration, t_P = 2/3 T_ds the time of the peak):
        F_ap(t) / F_ap0 = k1 exp(-((t - t_P) / T_ds)^2) - k2 t / T_ds,
        k1 = e^(4/9),  k2 = (k1 / 2) e^(-16/9).
    With tau = t / (2 T_ds) in [0, 1] (0 = contralateral heel strike, 1 = trailing toe-off):
        f(tau) = e^(4/9) exp(-(2 tau - 2/3)^2) - e^(-4/3) tau,
    so f(0) = 1 and f(1) = 0 exactly; the function rises to a peak of about 1.4745 at
    tau of about 0.31 and then decays to zero (the AP force of the trailing foot is not monotonic:
    push-off peaks early in double support).
    """
    tau = np.asarray(tau, float)
    return K1 * np.exp(-(2.0 * tau - 2.0 / 3.0) ** 2) - 2.0 * K2 * tau


SHAPES = {"ren_ap": sta_ap_shape}


def sta_split(S, FzR, FzL, fs, thr=None, shape="ren_ap", return_info=False):
    """STA: smooth transition assumption for the horizontal force.

    Source of the assumption: Ren, Jones & Howard (2008), "Whole body inverse dynamics over a
    complete gait cycle based only on measured kinematics", J Biomech 41(12):2750-2759,
    doi:10.1016/j.jbiomech.2008.06.001. The same assumption resolves the double-support
    indeterminacy in Lugris et al. (2013; method "STA": one law for the AP force,
    another for the other five components) and Mayo & Ojeda (2020;
    exponential-plus-linear laws, separate law for the non-monotonic AP force).
    The AP law used here (sta_ap_shape) is Ren's AP function: constants and form as reproduced in
    J. Morales Galera (2017), Univ. Sevilla TFG (hdl.handle.net/11441/65456), Sect. 3,
    Eqs. (9)-(10); its peak of 1.4745 F_ap0 matches the Ren curve plotted by Karatsidis et al.
    (2017), Sensors 17:75, Fig. 3 (anterior GRF, dashed). The exact constants in Lugris 2013 and
    Mayo & Ojeda 2020 were not checked against the originals.

    Rule, per double-support (DS) episode (both feet loaded, contact from THIS arm's own vertical
    force FzR, FzL > thr):
      * trailing foot = the foot that lifts off at the end of the episode (if the record ends
        inside the episode: the foot that was alone on the ground before it). In a regular
        episode this is also the foot that was alone on the ground before it;
      * F0 = S at the last single-support frame before the episode (the trailing foot's
        single-support value, given by the whole-body balance). Irregular episodes, where the
        foot that lifts was in the air before (a brief extra contact, e.g. a belt crossing),
        get F0 = 0 (its value before the episode), so the other foot keeps S and nothing jumps;
        if the episode starts the record or follows a flight frame, F0 = S at its first frame;
      * tau_i = (i - a + 1) / (L + 1) for the L frames i = a .. a+L-1 of the episode, i.e.
        t / T_DS with t measured from the last single-support frame and T_DS = (L + 1) / fs
        reaching the first frame of the next single support; so the trailing force is F0 at
        the last single-support frame and 0 at the first frame after the episode;
      * F_trail(i) = F0 * f(tau_i)   (f = sta_ap_shape by default),
        F_lead(i)  = S(i) - F_trail(i)   (the leading foot takes the remainder, so the balance
        holds exactly and the two feet can have opposite signs).
    Single support: the loaded foot takes S. Swing (and flight): 0.
    Episodes whose trailing foot cannot be identified (neither neighbour is single support)
    fall back to the proportional split and are counted in info["n_fallback"].

    fs only converts frames to seconds for info (the law is written on normalized time, so the
    estimate itself does not depend on fs).
    Returns FxR, FxL (and info if return_info).
    """
    S = np.asarray(S, float)
    n = len(S)
    onR, onL = contact(FzR, FzL, thr)
    f = SHAPES[shape]
    FxR = np.where(onR & ~onL, S, 0.0)
    FxL = np.where(onL & ~onR, S, 0.0)
    pR, pL = newton_prop(S, FzR, FzL)
    starts, ends = runs(onR & onL)
    on = {"R": onR, "L": onL}
    ep, n_fb, n_incons = [], 0, 0
    for a, b in zip(starts, ends):
        L = b - a
        before = ("R" if onR[a - 1] else "L") if (a > 0 and onR[a - 1] != onL[a - 1]) else None
        lifts = ("L" if onR[b] else "R") if (b < n and onR[b] != onL[b]) else None
        trail = lifts if lifts is not None else before
        if trail is None:
            FxR[a:b], FxL[a:b] = pR[a:b], pL[a:b]
            n_fb += 1
            ep.append(dict(start=int(a), end=int(b), trailing=None, F0=np.nan, dur_s=L / fs,
                           regular=False))
            continue
        # F0 = the trailing foot's value in the frame before the episode under the same rules:
        # S if it was alone on the ground, 0 if it was in the air (a brief extra contact), S[a] if
        # the episode starts the record or follows a flight frame
        if a > 0 and on[trail][a - 1] and before == trail:
            F0 = S[a - 1]
        elif a > 0 and not on[trail][a - 1] and before is not None:
            F0 = 0.0
        else:
            F0 = S[a]
        regular = before is not None and lifts is not None and before == lifts
        n_incons += int(before is not None and lifts is not None and before != lifts)
        tau = (np.arange(L) + 1.0) / (L + 1.0)
        Ft = F0 * f(tau)
        Fl = S[a:b] - Ft
        if trail == "R":
            FxR[a:b], FxL[a:b] = Ft, Fl
        else:
            FxL[a:b], FxR[a:b] = Ft, Fl
        ep.append(dict(start=int(a), end=int(b), trailing=trail, F0=float(F0), dur_s=(L + 1) / fs,
                       regular=bool(regular)))
    if return_info:
        return FxR, FxL, dict(episodes=ep, n_episodes=len(ep), n_fallback=n_fb,
                              n_inconsistent=n_incons)
    return FxR, FxL
