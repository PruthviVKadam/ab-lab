"""Monte-Carlo demonstration of the peeking (early-stopping) problem.

Pure functions, no UI. Under the null hypothesis (control and treatment share the
same true rate), a single fixed-horizon two-proportion test rejects at rate ~alpha.
But if you re-test at every "look" and stop as soon as p < alpha, the probability
of *ever* rejecting inflates well above alpha — that inflation is what these
simulations make visible.

All randomness is seeded, so results are reproducible and testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class PeekingResult:
    alpha: float
    n_sims: int
    look_sizes: np.ndarray        # per-group cumulative sample size at each look
    fpr_single: float             # P(reject) testing once at the final look
    fpr_peeking: float            # P(reject at ANY look) when you stop early
    example_pvalues: np.ndarray   # (n_examples, n_looks) running two-sided p-values


def _validate(true_rate, n_per_group, n_looks, alpha, n_sims):
    if not 0.0 < true_rate < 1.0:
        raise ValueError("true_rate must be in (0, 1).")
    if n_per_group < 2:
        raise ValueError("n_per_group must be >= 2.")
    if n_looks < 1:
        raise ValueError("n_looks must be >= 1.")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    if n_sims < 1:
        raise ValueError("n_sims must be >= 1.")


def _look_sizes(n_per_group: int, n_looks: int) -> np.ndarray:
    """Evenly spaced cumulative per-group sample sizes for `n_looks` interim analyses."""
    raw = np.linspace(n_per_group / n_looks, n_per_group, n_looks)
    looks = np.unique(np.round(raw).astype(int))
    looks = looks[looks >= 2]
    if looks.size == 0:
        looks = np.array([n_per_group])
    return looks


def _abs_z_pooled(succ_a: np.ndarray, succ_b: np.ndarray, m: int) -> np.ndarray:
    """Vectorized |z| for a pooled two-proportion test at per-group size m.

    Returns 0 where the pooled variance is undefined (both groups all-0 or all-1).
    """
    p_a = succ_a / m
    p_b = succ_b / m
    p_pool = (succ_a + succ_b) / (2.0 * m)
    var = p_pool * (1.0 - p_pool) * (2.0 / m)
    z = np.zeros_like(p_a, dtype=float)
    nz = var > 0
    z[nz] = (p_b[nz] - p_a[nz]) / np.sqrt(var[nz])
    return np.abs(z)


def _cumulative_successes(true_rate, n_per_group, n_sims, rng):
    a = (rng.random((n_sims, n_per_group)) < true_rate)
    b = (rng.random((n_sims, n_per_group)) < true_rate)
    cum_a = np.cumsum(a, axis=1, dtype=np.int32)
    cum_b = np.cumsum(b, axis=1, dtype=np.int32)
    return cum_a, cum_b


def run_peeking_simulation(
    true_rate: float = 0.20,
    n_per_group: int = 2000,
    n_looks: int = 10,
    alpha: float = 0.05,
    n_sims: int = 2000,
    n_examples: int = 25,
    seed: int = 0,
) -> PeekingResult:
    """Simulate `n_sims` null experiments and measure single-look vs peeking FPR."""
    _validate(true_rate, n_per_group, n_looks, alpha, n_sims)
    rng = np.random.default_rng(seed)
    cum_a, cum_b = _cumulative_successes(true_rate, n_per_group, n_sims, rng)

    looks = _look_sizes(n_per_group, n_looks)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    n_ex = min(n_examples, n_sims)
    example_p = np.empty((n_ex, looks.size))
    rejected_any = np.zeros(n_sims, dtype=bool)

    for j, m in enumerate(looks):
        abs_z = _abs_z_pooled(cum_a[:, m - 1], cum_b[:, m - 1], int(m))
        rejected_any |= abs_z > z_crit
        example_p[:, j] = 2.0 * stats.norm.sf(abs_z[:n_ex])

    final_abs_z = _abs_z_pooled(cum_a[:, looks[-1] - 1], cum_b[:, looks[-1] - 1], int(looks[-1]))
    fpr_single = float((final_abs_z > z_crit).mean())
    fpr_peeking = float(rejected_any.mean())

    return PeekingResult(
        alpha=alpha,
        n_sims=n_sims,
        look_sizes=looks,
        fpr_single=fpr_single,
        fpr_peeking=fpr_peeking,
        example_pvalues=example_p,
    )


def peeking_fpr_curve(
    true_rate: float = 0.20,
    n_per_group: int = 2000,
    max_looks: int = 20,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """False-positive rate as a function of the number of looks (1..max_looks).

    Generates one pool of null experiments and re-evaluates each look schedule on
    it, so the curve is monotonically non-decreasing in the number of looks.
    Returns (looks_axis, fprs).
    """
    _validate(true_rate, n_per_group, max_looks, alpha, n_sims)
    rng = np.random.default_rng(seed)
    cum_a, cum_b = _cumulative_successes(true_rate, n_per_group, n_sims, rng)
    z_crit = stats.norm.ppf(1 - alpha / 2)

    looks_axis = np.arange(1, max_looks + 1)
    fprs = np.empty(max_looks, dtype=float)
    for i, k in enumerate(looks_axis):
        rejected_any = np.zeros(n_sims, dtype=bool)
        for m in _look_sizes(n_per_group, int(k)):
            rejected_any |= _abs_z_pooled(cum_a[:, m - 1], cum_b[:, m - 1], int(m)) > z_crit
        fprs[i] = rejected_any.mean()
    return looks_axis, fprs
