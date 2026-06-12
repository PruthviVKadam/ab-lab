"""Aggregate tidy per-unit experiment data into binary success/trial counts.

Pure (pandas in, plain numbers out), no UI. Keeps the Streamlit layer free of
data-wrangling so the analysis is testable.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class BinarySummary:
    """Successes and trials per variant, ready for a two-proportion test."""

    control_label: str
    treatment_label: str
    successes_control: int
    n_control: int
    successes_treatment: int
    n_treatment: int

    @property
    def rate_control(self) -> float:
        return self.successes_control / self.n_control

    @property
    def rate_treatment(self) -> float:
        return self.successes_treatment / self.n_treatment


def available_groups(df: pd.DataFrame, group_col: str) -> list:
    """Sorted unique non-null values of the grouping column (for UI pickers)."""
    if group_col not in df.columns:
        raise ValueError(f"Column {group_col!r} not in dataframe.")
    return sorted(df[group_col].dropna().unique().tolist(), key=str)


def _to_binary(series: pd.Series) -> pd.Series:
    """Coerce an outcome column to 0/1 ints, or raise if it isn't binary.

    Accepts booleans, 0/1 numerics, and 2-value columns whose values are a
    subset of {0, 1, True, False}.
    """
    if series.isna().any():
        raise ValueError("Outcome column contains missing values.")
    if series.dtype == bool:
        return series.astype(int)
    unique = set(pd.unique(series))
    if not unique.issubset({0, 1, True, False}):
        raise ValueError(
            f"Outcome column must be binary (0/1 or True/False); found values {sorted(unique, key=str)}."
        )
    return series.astype(int)


def summarize_binary_experiment(
    df: pd.DataFrame,
    group_col: str,
    outcome_col: str,
    control_label,
    treatment_label,
) -> BinarySummary:
    """Reduce a tidy dataframe to (successes, n) for the control and treatment groups.

    Each row is one unit. `outcome_col` must be a binary success indicator.
    """
    for col in (group_col, outcome_col):
        if col not in df.columns:
            raise ValueError(f"Column {col!r} not in dataframe.")
    if control_label == treatment_label:
        raise ValueError("control_label and treatment_label must differ.")

    outcome = _to_binary(df[outcome_col])
    groups = df[group_col]

    control_mask = groups == control_label
    treatment_mask = groups == treatment_label
    n_control = int(control_mask.sum())
    n_treatment = int(treatment_mask.sum())
    if n_control == 0:
        raise ValueError(f"No rows for control group {control_label!r}.")
    if n_treatment == 0:
        raise ValueError(f"No rows for treatment group {treatment_label!r}.")

    return BinarySummary(
        control_label=str(control_label),
        treatment_label=str(treatment_label),
        successes_control=int(outcome[control_mask].sum()),
        n_control=n_control,
        successes_treatment=int(outcome[treatment_mask].sum()),
        n_treatment=n_treatment,
    )
