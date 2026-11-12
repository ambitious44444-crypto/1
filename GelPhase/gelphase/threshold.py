"""
Piecewise (broken-line) NLLS saturation threshold fitting.
"""
import numpy as np
from scipy.optimize import curve_fit
from gelphase.models import piecewise_linear


def fit_saturation_threshold(concentrations, sigmas, sigma_errs=None):
    """
    Fit a broken-line model to sigma vs. concentration data.

    Parameters
    ----------
    concentrations : array-like   — FBS concentration values (%)
    sigmas         : array-like   — extracted main-peak sigma values
    sigma_errs     : array-like   — optional sigma uncertainties (for weighting)

    Returns
    -------
    dict with keys:
        threshold, y_intercept, slope1, slope2, popt, x_dense, y_dense
    """
    c = np.asarray(concentrations, dtype=float)
    s = np.asarray(sigmas, dtype=float)

    # Initial guess: breakpoint at midrange
    c_mid = float(np.median(c))
    p0 = [c_mid, s[0], 0.1, (s[-1] - s[0]) / max(c[-1] - c[0], 1)]
    lo = [c.min(), 0.0, -5.0, 0.0]
    hi = [c.max(), s.max() * 2, 10.0, 50.0]

    sigma_weights = None
    if sigma_errs is not None:
        se = np.asarray(sigma_errs, dtype=float)
        se = np.where(se <= 0, 1e-6, se)
        sigma_weights = se

    popt, _ = curve_fit(
        piecewise_linear, c, s,
        p0=p0, bounds=(lo, hi),
        sigma=sigma_weights,
        absolute_sigma=(sigma_weights is not None),
        maxfev=20000
    )

    threshold    = float(popt[0])
    y_intercept  = float(popt[1])
    slope1       = float(popt[2])
    slope2       = float(popt[2] + popt[3])

    x_dense = np.linspace(c.min(), c.max(), 300)
    y_dense = piecewise_linear(x_dense, *popt)

    return {
        "threshold":   threshold,
        "y_intercept": y_intercept,
        "slope1":      slope1,
        "slope2":      slope2,
        "popt":        popt,
        "x_dense":     x_dense,
        "y_dense":     y_dense,
        "concentrations": c,
        "sigmas":      s,
        "sigma_errs":  sigma_errs,
    }
