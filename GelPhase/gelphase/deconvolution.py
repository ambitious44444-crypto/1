"""
Multi-Gaussian deconvolution with AICc model selection.
"""
import numpy as np
from scipy.optimize import curve_fit
from gelphase.models import gaussian, model_2peaks, model_3peaks, calc_aicc


def fit_gel_lane(x, y):
    """
    Fit a gel lane profile with 2-peak and 3-peak Gaussian models.
    Selects the best model by AICc (delta > 2 required to prefer 3-peak).

    Parameters
    ----------
    x : array-like  — pixel / distance axis
    y : array-like  — intensity axis

    Returns
    -------
    dict with keys:
        best_model, best_aic, main_mu, main_sigma, main_sigma_err,
        popt2, popt3, aic2, aic3, x, y
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Auto-detect peak regions
    x_min, x_max = x.min(), x.max()
    span = x_max - x_min

    # Main peak: first 40% of x range
    m1 = (x >= x_min + 0.05 * span) & (x <= x_min + 0.45 * span)
    if m1.sum() == 0:
        m1 = x <= np.median(x)
    p1_x = x[m1][np.argmax(y[m1])]
    p1_y = float(np.max(y[m1]))

    # Secondary peak: last 40% of x range
    m2 = (x >= x_min + 0.55 * span) & (x <= x_max - 0.05 * span)
    if m2.sum() == 0:
        m2 = x > np.median(x)
    p2_x = x[m2][np.argmax(y[m2])]
    p2_y = float(np.max(y[m2]))

    bg_init = float(np.mean(y[:max(1, len(y) // 20)]))
    sig_init = span * 0.05

    # ---- 2-peak model ----
    p0_2 = [-0.001, bg_init, p1_y, p1_x, sig_init,
             p2_y, p2_x, sig_init * 1.5]
    lo_2 = [-1.0, -abs(bg_init) * 3, 0.0, x_min, 1.0,
             0.0, x_min + span * 0.4, 1.0]
    hi_2 = [1.0, abs(bg_init) * 10 + 50, y.max() * 2, x_min + span * 0.5, span * 0.3,
             y.max() * 2, x_max, span * 0.4]
    try:
        popt2, pcov2 = curve_fit(model_2peaks, x, y, p0=p0_2,
                                 bounds=(lo_2, hi_2), maxfev=15000)
        aic2, _ = calc_aicc(y, model_2peaks(x, *popt2), len(popt2))
        err2 = np.sqrt(np.diag(pcov2))
    except Exception:
        popt2 = p0_2
        pcov2 = np.eye(len(p0_2))
        aic2 = 1e9
        err2 = np.ones(len(p0_2))

    # ---- 3-peak model ----
    p0_3 = [-0.001, bg_init, p1_y, p1_x, sig_init,
             p2_y, p2_x, sig_init * 1.5,
             p1_y * 0.3, p1_x - sig_init * 2, sig_init * 0.6]
    lo_3 = [-1.0, -abs(bg_init) * 3, 0.0, x_min, 1.0,
             0.0, x_min + span * 0.4, 1.0,
             0.0, x_min, 1.0]
    hi_3 = [1.0, abs(bg_init) * 10 + 50, y.max() * 2, x_min + span * 0.5, span * 0.3,
             y.max() * 2, x_max, span * 0.4,
             y.max(), x_min + span * 0.45, span * 0.15]
    try:
        popt3, pcov3 = curve_fit(model_3peaks, x, y, p0=p0_3,
                                 bounds=(lo_3, hi_3), maxfev=15000)
        aic3, _ = calc_aicc(y, model_3peaks(x, *popt3), len(popt3))
        err3 = np.sqrt(np.diag(pcov3))
    except Exception:
        popt3 = p0_3
        pcov3 = np.eye(len(p0_3))
        aic3 = 1e9
        err3 = np.ones(len(p0_3))

    # AICc model selection (prefer 3-peak only if delta > 2)
    use_3peak = (aic3 < aic2 - 2) and (aic3 < 1e8)
    if use_3peak:
        best_model = "3-Peak"
        best_aic = aic3
        main_mu = float(popt3[3])
        main_sigma = float(popt3[4])
        main_sigma_err = float(err3[4])
    else:
        best_model = "2-Peak"
        best_aic = aic2
        main_mu = float(popt2[3])
        main_sigma = float(popt2[4])
        main_sigma_err = float(err2[4])

    return {
        "best_model": best_model,
        "best_aic": best_aic,
        "main_mu": main_mu,
        "main_sigma": main_sigma,
        "main_sigma_err": main_sigma_err,
        "popt2": popt2,
        "popt3": popt3,
        "aic2": aic2,
        "aic3": aic3,
        "x": x,
        "y": y,
    }
