"""Hypothesis tests for A/B analysis.

Pure functions, no UI. The two-proportion z-test is implemented from first
principles (pooled variance under H0) so it can be unit-tested against the
statsmodels reference (``proportions_ztest``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

_VALID_ALTERNATIVES = {"two-sided", "larger", "smaller"}


@dataclass(frozen=True)
class ABResult:
    """Outcome of a two-proportion comparison (B vs A).

    H0: p_b == p_a.  H1 depends on `alternative`.
    Assumptions: two independent groups, binary outcome, fixed-horizon read,
    normal approximation valid (n*p and n*(1-p) >= ~10 per group).
    """

    rate_a: float
    rate_b: float
    abs_diff: float           # p_b - p_a
    rel_lift: float           # (p_b - p_a) / p_a
    z_stat: float
    p_value: float
    ci_low: float             # CI on the absolute difference (unpooled SE)
    ci_high: float
    alpha: float
    alternative: str

    @property
    def significant(self) -> bool:
        return self.p_value < self.alpha

    def verdict(self) -> str:
        """Plain-English call for the UI."""
        if not self.significant:
            return "Not statistically significant — do not ship on this evidence."
        direction = "increase" if self.abs_diff > 0 else "decrease"
        return f"Statistically significant {direction} (p={self.p_value:.4f})."


def two_proportion_ztest(
    successes_a: int,
    n_a: int,
    successes_b: int,
    n_b: int,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> ABResult:
    """Pooled two-proportion z-test comparing variant B against control A.

    The test statistic uses the POOLED proportion (the variance under H0). The
    confidence interval on the difference uses the UNPOOLED standard error
    (the standard convention), so a p<alpha result and a CI excluding 0 agree.
    """
    if alternative not in _VALID_ALTERNATIVES:
        raise ValueError(f"alternative must be one of {_VALID_ALTERNATIVES}, got {alternative!r}.")
    if n_a <= 0 or n_b <= 0:
        raise ValueError("n_a and n_b must be positive.")
    if not 0 <= successes_a <= n_a or not 0 <= successes_b <= n_b:
        raise ValueError("successes must satisfy 0 <= successes <= n for each group.")

    p_a = successes_a / n_a
    p_b = successes_b / n_b
    p_pool = (successes_a + successes_b) / (n_a + n_b)

    se_pooled = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_pooled == 0:
        raise ValueError("Degenerate input: pooled standard error is zero (rates are 0 or 1 in both groups).")

    z = (p_b - p_a) / se_pooled

    if alternative == "two-sided":
        p_value = 2 * stats.norm.sf(abs(z))
    elif alternative == "larger":      # H1: p_b > p_a
        p_value = stats.norm.sf(z)
    else:                              # "smaller", H1: p_b < p_a
        p_value = stats.norm.cdf(z)

    # CI on absolute difference, unpooled SE.
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    z_crit = stats.norm.ppf(1 - alpha / 2)
    diff = p_b - p_a
    ci_low = diff - z_crit * se_unpooled
    ci_high = diff + z_crit * se_unpooled

    rel_lift = (diff / p_a) if p_a > 0 else float("nan")

    return ABResult(
        rate_a=p_a,
        rate_b=p_b,
        abs_diff=diff,
        rel_lift=rel_lift,
        z_stat=z,
        p_value=float(p_value),
        ci_low=ci_low,
        ci_high=ci_high,
        alpha=alpha,
        alternative=alternative,
    )
