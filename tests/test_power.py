"""Tests for stats.power.

Strategy: validate the statsmodels-based implementation against an INDEPENDENT
textbook normal-approximation formula (different math than the Cohen's-h path
statsmodels uses), plus monotonicity and self-consistency properties.
"""

import math

import pytest
from scipy import stats as sps

from stats.power import (
    effect_size,
    power_for_sample_size,
    required_sample_size,
    sensitivity_curve,
)


def textbook_sample_size(p1, mde, alpha=0.05, power=0.80):
    """Independent oracle: standard two-proportion sample-size formula (per group)."""
    p2 = p1 + mde
    p_bar = (p1 + p2) / 2
    z_alpha = sps.norm.ppf(1 - alpha / 2)
    z_beta = sps.norm.ppf(power)
    numerator = (
        z_alpha * math.sqrt(2 * p_bar * (1 - p_bar))
        + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return numerator / (mde ** 2)


def test_matches_textbook_formula_within_10pct():
    # Two independent methods should land close to each other.
    for p1, mde in [(0.10, 0.02), (0.20, 0.05), (0.05, 0.01)]:
        got = required_sample_size(p1, mde, alpha=0.05, power=0.80)
        oracle = textbook_sample_size(p1, mde, alpha=0.05, power=0.80)
        assert abs(got - oracle) / oracle < 0.10, (p1, mde, got, oracle)


def test_smaller_mde_needs_more_samples():
    big = required_sample_size(0.10, 0.05)
    small = required_sample_size(0.10, 0.01)
    assert small > big


def test_higher_power_needs_more_samples():
    low = required_sample_size(0.10, 0.02, power=0.80)
    high = required_sample_size(0.10, 0.02, power=0.95)
    assert high > low


def test_lower_alpha_needs_more_samples():
    lax = required_sample_size(0.10, 0.02, alpha=0.10)
    strict = required_sample_size(0.10, 0.02, alpha=0.01)
    assert strict > lax


def test_power_at_required_n_meets_target():
    # Solving for n then computing power back should recover >= the target
    # (>= because we round n up).
    target = 0.80
    n = required_sample_size(0.10, 0.02, alpha=0.05, power=target)
    achieved = power_for_sample_size(0.10, 0.02, n, alpha=0.05)
    assert achieved >= target


def test_effect_size_is_absolute_and_symmetric():
    up = effect_size(0.10, 0.02)
    down = effect_size(0.12, -0.02)
    assert up == pytest.approx(down, rel=1e-9)
    assert up > 0


def test_sensitivity_curve_is_decreasing():
    mde_grid, sizes = sensitivity_curve(0.10, 0.005, 0.05, n_points=15)
    assert len(mde_grid) == len(sizes) == 15
    # Smaller effects -> larger samples, so the series is non-increasing in MDE.
    assert all(sizes[i] >= sizes[i + 1] for i in range(len(sizes) - 1))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"baseline_rate": 0.0, "mde_absolute": 0.02},
        {"baseline_rate": 1.0, "mde_absolute": 0.02},
        {"baseline_rate": 0.99, "mde_absolute": 0.02},  # treatment rate > 1
        {"baseline_rate": 0.10, "mde_absolute": 0.0},
        {"baseline_rate": 0.10, "mde_absolute": 0.02, "alpha": 0.0},
        {"baseline_rate": 0.10, "mde_absolute": 0.02, "power": 1.0},
        {"baseline_rate": 0.10, "mde_absolute": 0.02, "alternative": "bogus"},
    ],
)
def test_invalid_inputs_raise(kwargs):
    with pytest.raises(ValueError):
        required_sample_size(**kwargs)
