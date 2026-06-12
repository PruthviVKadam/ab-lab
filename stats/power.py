"""Power analysis for two-proportion A/B tests.

All functions are pure (no I/O, no Streamlit). They wrap statsmodels'
``NormalIndPower`` using Cohen's h as the effect size for proportions, so results
are verifiable against the statsmodels reference implementation.

Definitions
-----------
- baseline_rate (p1): control conversion rate, in (0, 1).
- mde_absolute: minimum detectable effect, in ABSOLUTE proportion points
  (e.g. 0.02 means detect a lift from 0.10 to 0.12).
- alpha: false-positive rate (significance level).
- power (1 - beta): probability of detecting a true effect of size MDE.

Assumptions / when these are INVALID
------------------------------------
- Two independent groups, binary outcome per unit, fixed-horizon test
  (results read once at the planned sample size — NOT during peeking).
- The normal approximation to the binomial is used; it degrades for very small
  n or for rates extremely close to 0 or 1. Prefer n*p and n*(1-p) >= ~10.
"""

from __future__ import annotations

import numpy as np
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize

_VALID_ALTERNATIVES = {"two-sided", "larger", "smaller"}


def _validate_rates(baseline_rate: float, treatment_rate: float) -> None:
    for name, value in (("baseline_rate", baseline_rate), ("treatment_rate", treatment_rate)):
        if not 0.0 < value < 1.0:
            raise ValueError(f"{name} must be strictly between 0 and 1, got {value}.")


def effect_size(baseline_rate: float, mde_absolute: float) -> float:
    """Cohen's h between the treatment and baseline rates.

    Returns the absolute effect size used by the power calculations.
    """
    treatment_rate = baseline_rate + mde_absolute
    _validate_rates(baseline_rate, treatment_rate)
    # proportion_effectsize(prop1, prop2) = 2*arcsin(sqrt(prop1)) - 2*arcsin(sqrt(prop2))
    return abs(proportion_effectsize(treatment_rate, baseline_rate))


def required_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> int:
    """Required sample size PER variant (control group; treatment = ratio * this).

    Returns an integer count, rounded up.
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}.")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}.")
    if mde_absolute == 0:
        raise ValueError("mde_absolute must be non-zero; you cannot power for a zero effect.")

    h = effect_size(baseline_rate, mde_absolute)
    n = NormalIndPower().solve_power(
        effect_size=h,
        alpha=alpha,
        power=power,
        ratio=ratio,
        alternative=alternative,
    )
    return int(np.ceil(n))


def power_for_sample_size(
    baseline_rate: float,
    mde_absolute: float,
    n_per_group: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
    ratio: float = 1.0,
) -> float:
    """Achieved power for a given per-group sample size. Returns a value in (0, 1)."""
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}.")
    if n_per_group <= 0:
        raise ValueError(f"n_per_group must be positive, got {n_per_group}.")

    h = effect_size(baseline_rate, mde_absolute)
    return float(
        NormalIndPower().power(
            effect_size=h,
            nobs1=n_per_group,
            alpha=alpha,
            ratio=ratio,
            alternative=alternative,
        )
    )


def sensitivity_curve(
    baseline_rate: float,
    mde_min: float,
    mde_max: float,
    alpha: float = 0.05,
    power: float = 0.80,
    alternative: str = "two-sided",
    n_points: int = 40,
) -> tuple[np.ndarray, np.ndarray]:
    """Required per-group sample size as a function of MDE.

    Returns (mde_grid, sample_sizes). Smaller effects need more samples, so the
    curve is monotonically decreasing in MDE — useful to show the cost of
    detecting subtle effects.
    """
    if mde_min <= 0 or mde_max <= 0:
        raise ValueError("mde_min and mde_max must be positive (absolute proportion points).")
    if mde_min >= mde_max:
        raise ValueError("mde_min must be less than mde_max.")

    mde_grid = np.linspace(mde_min, mde_max, n_points)
    sizes = np.array(
        [
            required_sample_size(baseline_rate, float(mde), alpha=alpha, power=power, alternative=alternative)
            for mde in mde_grid
        ]
    )
    return mde_grid, sizes
