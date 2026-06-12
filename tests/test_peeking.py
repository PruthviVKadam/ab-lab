"""Tests for stats.peeking.

The key statistical facts under the null (control rate == treatment rate):
- testing once gives a false-positive rate ~ alpha,
- peeking at many looks inflates it well above alpha,
- the inflation grows monotonically with the number of looks.
"""

import numpy as np
import pytest

from stats.peeking import peeking_fpr_curve, run_peeking_simulation


def test_single_look_fpr_is_about_alpha():
    # n_looks=1 means one test at the end: false-positive rate should be ~alpha.
    r = run_peeking_simulation(true_rate=0.2, n_per_group=1500, n_looks=1, alpha=0.05, n_sims=4000, seed=1)
    assert r.fpr_single == pytest.approx(0.05, abs=0.02)
    # With a single look, "peeking" and "single" are the same test.
    assert r.fpr_peeking == pytest.approx(r.fpr_single, abs=1e-9)


def test_peeking_inflates_false_positive_rate():
    r = run_peeking_simulation(true_rate=0.2, n_per_group=1500, n_looks=20, alpha=0.05, n_sims=4000, seed=2)
    # Peeking at 20 looks rejects far more often than a single look under the null.
    assert r.fpr_peeking > r.fpr_single
    assert r.fpr_peeking > 0.10            # materially above the nominal 0.05
    assert r.fpr_single == pytest.approx(0.05, abs=0.02)


def test_fpr_curve_is_monotone_nondecreasing_and_grows():
    looks, fprs = peeking_fpr_curve(
        true_rate=0.2, n_per_group=1500, max_looks=15, alpha=0.05, n_sims=4000, seed=3
    )
    assert looks.tolist() == list(range(1, 16))
    # Same data pool re-evaluated -> more looks can only add rejections.
    assert np.all(np.diff(fprs) >= -1e-12)
    assert fprs[0] == pytest.approx(0.05, abs=0.02)
    assert fprs[-1] > fprs[0]


def test_determinism_same_seed():
    kw = dict(true_rate=0.3, n_per_group=800, n_looks=8, alpha=0.05, n_sims=1000, seed=7)
    a = run_peeking_simulation(**kw)
    b = run_peeking_simulation(**kw)
    assert a.fpr_peeking == b.fpr_peeking
    assert a.fpr_single == b.fpr_single
    assert np.array_equal(a.example_pvalues, b.example_pvalues)


def test_example_pvalues_shape_and_range():
    r = run_peeking_simulation(true_rate=0.25, n_per_group=1000, n_looks=10, n_sims=500, n_examples=12, seed=4)
    assert r.example_pvalues.shape == (12, r.look_sizes.size)
    assert r.example_pvalues.min() >= 0.0
    assert r.example_pvalues.max() <= 1.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"true_rate": 0.0},
        {"true_rate": 1.0},
        {"n_per_group": 1},
        {"n_looks": 0},
        {"alpha": 0.0},
        {"n_sims": 0},
    ],
)
def test_invalid_inputs_raise(kwargs):
    base = dict(true_rate=0.2, n_per_group=100, n_looks=5, alpha=0.05, n_sims=100)
    base.update(kwargs)
    with pytest.raises(ValueError):
        run_peeking_simulation(**base)
