"""
Core mathematical models for gel lane deconvolution and threshold fitting.
"""
import numpy as np


def gaussian(x, amp, mu, sigma):
    """Single Gaussian peak."""
    return amp * np.exp(-((x - mu) ** 2) / (2 * (sigma ** 2)))


def model_2peaks(x, bg_a, bg_b, A1, mu1, sig1, A2, mu2, sig2):
    """Linear baseline + 2 Gaussian peaks."""
    return bg_a * x + bg_b + gaussian(x, A1, mu1, sig1) + gaussian(x, A2, mu2, sig2)


def model_3peaks(x, bg_a, bg_b, A1, mu1, sig1, A2, mu2, sig2, A3, mu3, sig3):
    """Linear baseline + 3 Gaussian peaks."""
    return (bg_a * x + bg_b
            + gaussian(x, A1, mu1, sig1)
            + gaussian(x, A2, mu2, sig2)
            + gaussian(x, A3, mu3, sig3))


def calc_aicc(y_true, y_pred, k):
    """Corrected AIC (AICc). Returns (aicc, rss)."""
    n = len(y_true)
    rss = np.sum((y_true - y_pred) ** 2)
    aic = n * np.log(rss / n) + 2 * k
    if n - k - 1 > 0:
        aic += (2 * k * (k + 1)) / (n - k - 1)
    return aic, rss


def piecewise_linear(x, x_break, y_intercept, slope1, delta_slope):
    """
    Broken-line model for threshold detection.
    sigma(c) = y_intercept + slope1*c + delta_slope * max(0, c - x_break)
    """
    return y_intercept + slope1 * x + delta_slope * np.maximum(0, x - x_break)
