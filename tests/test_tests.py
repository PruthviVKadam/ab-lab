"""Tests for stats.tests.two_proportion_ztest.

Validated against (a) a fully hand-computed example and (b) the statsmodels
reference implementation.
"""

import pytest
from statsmodels.stats.proportion import proportion_confint, proportions_ztest

from stats.tests import proportion_ci, two_proportion_ztest


def test_hand_computed_example():
    # A: 200/1000 = 0.20, B: 240/1000 = 0.24
    # p_pool = 0.22, se = sqrt(0.22*0.78*(2/1000)) = 0.01852566
    # z = 0.04 / 0.01852566 = 2.1591676 ; two-sided p = 2*norm.sf(z) = 0.0308372
    r = two_proportion_ztest(200, 1000, 240, 1000)
    assert r.rate_a == pytest.approx(0.20)
    assert r.rate_b == pytest.approx(0.24)
    assert r.z_stat == pytest.approx(2.1591676, abs=1e-6)
    assert r.p_value == pytest.approx(0.0308372, abs=1e-6)
    assert r.abs_diff == pytest.approx(0.04)
    assert r.rel_lift == pytest.approx(0.20)
    assert r.significant is True
    assert r.ci_low > 0  # CI on the difference excludes 0 when significant


def test_matches_statsmodels_reference():
    # statsmodels uses the pooled proportion when the null value is 0,
    # matching our pooled z-statistic and two-sided p-value.
    sm_z, sm_p = proportions_ztest(count=[240, 200], nobs=[1000, 1000])
    r = two_proportion_ztest(200, 1000, 240, 1000)
    assert r.z_stat == pytest.approx(sm_z, abs=1e-9)
    assert r.p_value == pytest.approx(sm_p, abs=1e-9)


def test_one_sided_alternatives():
    larger = two_proportion_ztest(200, 1000, 240, 1000, alternative="larger")
    smaller = two_proportion_ztest(200, 1000, 240, 1000, alternative="smaller")
    two_sided = two_proportion_ztest(200, 1000, 240, 1000, alternative="two-sided")
    # For a positive effect: one-sided "larger" p is half the two-sided p.
    assert larger.p_value == pytest.approx(two_sided.p_value / 2, abs=1e-9)
    # "larger" and "smaller" p-values sum to 1 (continuous distribution).
    assert larger.p_value + smaller.p_value == pytest.approx(1.0, abs=1e-9)


def test_no_difference_is_not_significant():
    r = two_proportion_ztest(150, 1000, 150, 1000)
    assert r.z_stat == pytest.approx(0.0, abs=1e-12)
    assert r.p_value == pytest.approx(1.0, abs=1e-9)
    assert r.significant is False
    assert "Not statistically significant" in r.verdict()


def test_ci_brackets_point_difference():
    r = two_proportion_ztest(200, 1000, 240, 1000)
    assert r.ci_low < r.abs_diff < r.ci_high


@pytest.mark.parametrize(
    "args",
    [
        (200, 0, 240, 1000),       # n_a = 0
        (200, 1000, 240, 0),       # n_b = 0
        (1200, 1000, 240, 1000),   # successes > n
        (-1, 1000, 240, 1000),     # negative successes
    ],
)
def test_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        two_proportion_ztest(*args)


def test_invalid_alternative_raises():
    with pytest.raises(ValueError):
        two_proportion_ztest(200, 1000, 240, 1000, alternative="nope")


def test_proportion_ci_matches_statsmodels_normal():
    lo, hi = proportion_ci(240, 1000, alpha=0.05)
    sm_lo, sm_hi = proportion_confint(240, 1000, alpha=0.05, method="normal")
    assert lo == pytest.approx(sm_lo, abs=1e-9)
    assert hi == pytest.approx(sm_hi, abs=1e-9)


def test_proportion_ci_is_clipped_to_unit_interval():
    lo, hi = proportion_ci(0, 50)            # p_hat = 0 -> SE 0, low must not go negative
    assert lo == 0.0
    lo2, hi2 = proportion_ci(50, 50)         # p_hat = 1 -> high must not exceed 1
    assert hi2 == 1.0


@pytest.mark.parametrize("args", [(5, 0), (-1, 10), (11, 10)])
def test_proportion_ci_invalid_inputs_raise(args):
    with pytest.raises(ValueError):
        proportion_ci(*args)
