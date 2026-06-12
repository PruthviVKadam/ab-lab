"""Tests for stats.cuped."""

import numpy as np
import pytest

from stats.cuped import cuped_adjust, cuped_demo, variance_reduction


def test_cuped_preserves_mean_and_reduces_variance():
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, size=5000)
    y = 2.0 * x + rng.normal(0, 1, size=5000)  # y strongly correlated with x

    y_cv, theta = cuped_adjust(y, x)

    # Mean is preserved (unbiased), variance is reduced.
    assert y_cv.mean() == pytest.approx(y.mean(), abs=1e-9)
    assert y_cv.var(ddof=1) < y.var(ddof=1)
    # theta ~ cov/var ~ 2.0 for this generative model.
    assert theta == pytest.approx(2.0, abs=0.1)


def test_variance_reduction_equals_corr_squared():
    rng = np.random.default_rng(7)
    x = rng.normal(size=2000)
    y = 1.5 * x + rng.normal(size=2000)

    corr = np.corrcoef(y, x)[0, 1]
    assert variance_reduction(y, x) == pytest.approx(corr ** 2, rel=1e-9)

    # The realized reduction matches the theoretical factor (1 - corr^2).
    y_cv, _ = cuped_adjust(y, x)
    realized = 1 - y_cv.var(ddof=1) / y.var(ddof=1)
    assert realized == pytest.approx(corr ** 2, abs=0.02)


def test_uncorrelated_covariate_barely_reduces():
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    y = rng.normal(size=5000)  # independent of x
    assert variance_reduction(y, x) < 0.01


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        cuped_adjust(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_zero_variance_covariate_raises():
    with pytest.raises(ValueError):
        cuped_adjust(np.array([1.0, 2.0, 3.0]), np.array([5.0, 5.0, 5.0]))


def test_cuped_demo_shrinks_se_by_sqrt_one_minus_rho_squared():
    rho = 0.6
    d = cuped_demo(rho, n_users=40000, true_effect=0.1, seed=0)
    # CUPED leaves the estimate ~unbiased...
    assert d.raw_estimate == pytest.approx(0.1, abs=0.03)
    assert d.cuped_estimate == pytest.approx(0.1, abs=0.03)
    # ...and shrinks the SE by ~sqrt(1 - rho^2).
    assert d.se_ratio == pytest.approx(np.sqrt(1 - rho**2), abs=0.03)
    assert d.cuped_se < d.raw_se
    assert d.reduction == pytest.approx(rho**2, abs=0.02)
    assert d.var_y_cuped < d.var_y


def test_cuped_demo_zero_correlation_no_reduction():
    d = cuped_demo(0.0, n_users=20000, true_effect=0.1, seed=1)
    assert d.reduction < 0.01
    assert d.se_ratio == pytest.approx(1.0, abs=0.03)


@pytest.mark.parametrize("kwargs", [{"rho": 1.0}, {"rho": -0.1}, {"n_users": 2}])
def test_cuped_demo_invalid_inputs_raise(kwargs):
    base = dict(rho=0.5, n_users=1000)
    base.update(kwargs)
    with pytest.raises(ValueError):
        cuped_demo(**base)
