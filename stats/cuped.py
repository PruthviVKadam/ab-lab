"""CUPED (Controlled-experiment Using Pre-Experiment Data) variance reduction.

Pure function, no UI. CUPED removes the part of the outcome that is predictable
from a pre-experiment covariate, shrinking the metric's variance WITHOUT biasing
the treatment-effect estimate (the covariate must be pre-treatment, so treatment
cannot affect it).

Reference: Deng, Xu, Kohavi, Walker (2013), "Improving the Sensitivity of Online
Controlled Experiments by Utilizing Pre-Experiment Data."
"""

from __future__ import annotations

from dataclasses import dataclass

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


@dataclass(frozen=True)
class CupedDemo:
    """A simulated experiment analyzed with and without CUPED (for the UI demo)."""

    rho_target: float          # requested corr(pre-metric, outcome)
    rho_actual: float          # realized corr(x, y) that CUPED actually exploits
    true_effect: float
    raw_estimate: float        # difference in means of the raw metric
    raw_se: float
    cuped_estimate: float      # difference in means of the CUPED-adjusted metric
    cuped_se: float
    var_y: float
    var_y_cuped: float
    reduction: float           # corr(x, y)**2

    @property
    def se_ratio(self) -> float:
        return self.cuped_se / self.raw_se


def _diff_in_means_se(values: np.ndarray, treated: np.ndarray) -> tuple[float, float]:
    t = values[treated]
    c = values[~treated]
    est = float(t.mean() - c.mean())
    se = float(np.sqrt(t.var(ddof=1) / t.size + c.var(ddof=1) / c.size))
    return est, se


def cuped_demo(rho: float, n_users: int, true_effect: float = 0.1, seed: int = 0) -> CupedDemo:
    """Simulate a balanced A/B experiment with a pre-experiment covariate of correlation `rho`.

    Builds an outcome `y` correlated with a pre-metric `x` at strength `rho`, adds a
    treatment effect `true_effect` to a random half, then estimates that effect twice:
    once on the raw metric and once on the CUPED-adjusted metric. CUPED leaves the
    estimate unbiased but shrinks the standard error by ~sqrt(1 - rho**2).
    """
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must be in [0, 1).")
    if n_users < 4:
        raise ValueError("n_users must be >= 4.")

    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n_users)
    noise = rng.standard_normal(n_users)
    y0 = rho * x + np.sqrt(1.0 - rho**2) * noise   # corr(x, y0) ~ rho, var ~ 1

    treated = rng.permutation(n_users) < (n_users // 2)   # exactly n//2 treated, randomly placed
    y = y0 + true_effect * treated

    raw_estimate, raw_se = _diff_in_means_se(y, treated)
    y_cv, _ = cuped_adjust(y, x)
    cuped_estimate, cuped_se = _diff_in_means_se(y_cv, treated)

    return CupedDemo(
        rho_target=rho,
        rho_actual=float(np.corrcoef(x, y)[0, 1]),
        true_effect=true_effect,
        raw_estimate=raw_estimate,
        raw_se=raw_se,
        cuped_estimate=cuped_estimate,
        cuped_se=cuped_se,
        var_y=float(np.var(y, ddof=1)),
        var_y_cuped=float(np.var(y_cv, ddof=1)),
        reduction=variance_reduction(y, x),
    )
