"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Pure function, no UI. CUPED removes the part of the outcome that is predictable
from a pre-experiment covariate, shrinking the metric's variance WITHOUT biasing
the treatment-effect estimate (the covariate must be pre-treatment, so treatment
cannot affect it).

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data."
"""

from __future__ import annotations

import numpy as np


def cuped_adjust(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the CUPED-adjusted outcome and the fitted theta.

    Parameters
    ----------
    y : array of in-experiment metric values.
    x : array of PRE-experiment covariate values (same units, same subjects).
        Must be measured before treatment assignment, or CUPED is invalid.

    Returns
    -------
    (y_cv, theta) where ``y_cv = y - theta * (x - mean(x))`` and
    ``theta = cov(y, x) / var(x)``. ``y_cv`` has the same mean as ``y`` but
    variance reduced by a factor of (1 - corr(x, y)**2).
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    if y.shape != x.shape:
        raise ValueError(f"y and x must have the same shape, got {y.shape} and {x.shape}.")
    if y.size < 2:
        raise ValueError("Need at least 2 observations to estimate theta.")

    var_x = np.var(x, ddof=1)
    if var_x == 0:
        raise ValueError("Covariate x has zero variance; CUPED cannot reduce variance.")

    cov_xy = np.cov(y, x, ddof=1)[0, 1]
    theta = cov_xy / var_x
    y_cv = y - theta * (x - x.mean())
    return y_cv, float(theta)


def variance_reduction(y: np.ndarray, x: np.ndarray) -> float:
    """Fraction of variance removed by CUPED, equal to corr(x, y)**2 (in [0, 1))."""
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    corr = np.corrcoef(y, x)[0, 1]
    return float(corr ** 2)
