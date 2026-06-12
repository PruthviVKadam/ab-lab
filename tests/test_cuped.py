"""Tests for stats.cuped."""

import numpy as np
import pytest

from stats.cuped import cuped_adjust, variance_reduction


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
