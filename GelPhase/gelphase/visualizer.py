"""
Publication-quality matplotlib visualizations (Agg backend, cloud-safe).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gelphase.models import gaussian, model_2peaks, model_3peaks


def plot_spectral_decomposition(fit_res, title="Gel Lane Deconvolution"):
    """
    Plot raw data + 2-peak and 3-peak fits with individual Gaussian components.
    """
    x = fit_res["x"]
    y = fit_res["y"]
    popt2 = fit_res["popt2"]
    popt3 = fit_res["popt3"]
    aic2 = fit_res["aic2"]
    aic3 = fit_res["aic3"]
    best = fit_res["best_model"]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=130)
    ax.scatter(x, y, color="black", s=8, alpha=0.45, label="Raw Data", zorder=3)

    # 2-peak fit
    if aic2 < 1e8:
        bg2 = popt2[0] * x + popt2[1]
        p1_2 = gaussian(x, popt2[2], popt2[3], popt2[4])
        p2_2 = gaussian(x, popt2[5], popt2[6], popt2[7])
        ax.plot(x, model_2peaks(x, *popt2), "b-", lw=1.8, alpha=0.85,
                label=f"2-Peak fit  (AICc={aic2:.0f})")
        ax.fill_between(x, bg2, bg2 + p1_2, alpha=0.18, color="steelblue",
                        label=f"Peak 1 (σ={popt2[4]:.2f})")
        ax.fill_between(x, bg2, bg2 + p2_2, alpha=0.18, color="cornflowerblue",
                        label=f"Peak 2 (σ={popt2[7]:.2f})")

    # 3-peak fit
    if aic3 < 1e8:
        ax.plot(x, model_3peaks(x, *popt3), "r--", lw=1.8, alpha=0.85,
                label=f"3-Peak fit  (AICc={aic3:.0f})")

    ax.set_title(f"{title}\n[Best: {best}]", fontsize=11, fontweight="bold")
    ax.set_xlabel("Channel (pixel)", fontsize=10)
    ax.set_ylabel("Intensity (a.u.)", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_threshold_fit(th_res, title="Saturation Threshold Fit"):
    """
    Plot sigma vs. concentration with broken-line NLLS fit.
    """
    c = th_res["concentrations"]
    s = th_res["sigmas"]
    se = th_res.get("sigma_errs")
    x_dense = th_res["x_dense"]
    y_dense = th_res["y_dense"]
    threshold = th_res["threshold"]
    slope1 = th_res["slope1"]
    slope2 = th_res["slope2"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=130)

    if se is not None:
        ax.errorbar(c, s, yerr=se, fmt="o", color="darkblue",
                    markersize=7, capsize=5, label=r"Extracted $\sigma_{main}$", zorder=4)
    else:
        ax.scatter(c, s, color="darkblue", s=55, zorder=4,
                   label=r"Extracted $\sigma_{main}$")

    ax.plot(x_dense, y_dense, "r-", lw=2.2,
            label=(f"Piecewise NLLS fit\n"
                   f"$C_{{th}}$ = {threshold:.2f}%\n"
                   f"Slope₁ = {slope1:.3f},  Slope₂ = {slope2:.3f}"))
    ax.axvline(threshold, color="gray", linestyle="--", lw=1.5,
               label=f"Breakpoint = {threshold:.2f}%")

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Concentration (%)", fontsize=11)
    ax.set_ylabel(r"Main Peak Width ($\sigma_{main}$)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, linestyle=":", alpha=0.55)
    fig.tight_layout()
    return fig
