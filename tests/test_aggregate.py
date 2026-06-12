"""Tests for stats.aggregate, including a real-data sanity check on Cookie Cats."""

from pathlib import Path

import pandas as pd
import pytest

from stats.aggregate import available_groups, summarize_binary_experiment

DATA = Path(__file__).resolve().parents[1] / "data" / "cookie_cats.csv"


def _toy():
    # 3 control rows (1 success), 2 treatment rows (2 successes)
    return pd.DataFrame(
        {
            "grp": ["A", "A", "A", "B", "B"],
            "won": [True, False, False, True, True],
        }
    )


def test_summarize_counts_bool_outcome():
    s = summarize_binary_experiment(_toy(), "grp", "won", "A", "B")
    assert (s.successes_control, s.n_control) == (1, 3)
    assert (s.successes_treatment, s.n_treatment) == (2, 2)
    assert s.rate_control == pytest.approx(1 / 3)
    assert s.rate_treatment == pytest.approx(1.0)


def test_summarize_accepts_0_1_integers():
    df = pd.DataFrame({"grp": ["A", "A", "B", "B"], "won": [1, 0, 1, 1]})
    s = summarize_binary_experiment(df, "grp", "won", "A", "B")
    assert (s.successes_control, s.n_control, s.successes_treatment, s.n_treatment) == (1, 2, 2, 2)


def test_available_groups_sorted_unique():
    assert available_groups(_toy(), "grp") == ["A", "B"]


def test_non_binary_outcome_raises():
    df = pd.DataFrame({"grp": ["A", "B"], "won": [3, 7]})
    with pytest.raises(ValueError):
        summarize_binary_experiment(df, "grp", "won", "A", "B")


def test_missing_values_raise():
    df = pd.DataFrame({"grp": ["A", "B"], "won": [True, None]})
    with pytest.raises(ValueError):
        summarize_binary_experiment(df, "grp", "won", "A", "B")


def test_missing_column_raises():
    with pytest.raises(ValueError):
        summarize_binary_experiment(_toy(), "grp", "nope", "A", "B")


def test_same_labels_raise():
    with pytest.raises(ValueError):
        summarize_binary_experiment(_toy(), "grp", "won", "A", "A")


def test_absent_group_label_raises():
    with pytest.raises(ValueError):
        summarize_binary_experiment(_toy(), "grp", "won", "A", "Z")


@pytest.mark.skipif(not DATA.exists(), reason="cookie_cats.csv not present")
def test_cookie_cats_real_data_shape_and_counts():
    df = pd.read_csv(DATA)
    assert df.shape == (90189, 5)
    s = summarize_binary_experiment(df, "version", "retention_7", "gate_30", "gate_40")
    # Group sizes are fixed properties of the published dataset.
    assert s.n_control == 44700
    assert s.n_treatment == 45489
    # 7-day retention is higher in control (gate_30) than treatment (gate_40).
    assert s.rate_control > s.rate_treatment
