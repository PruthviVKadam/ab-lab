"""Experimentation Lab — A/B Test Designer & Analyzer.

UI only. All statistics live in the `stats/` package (pure, unit-tested).
Tabs: Design (live) · Analyze (live) · Pitfalls (roadmap).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stats.aggregate import available_groups, summarize_binary_experiment
from stats.power import power_for_sample_size, required_sample_size, sensitivity_curve
from stats.tests import proportion_ci, two_proportion_ztest

st.set_page_config(page_title="Experimentation Lab", page_icon="🧪", layout="wide")

st.title("🧪 Experimentation Lab")
st.caption("Design and analyze A/B tests — power analysis, peeking pitfalls, and variance reduction.")

DATA_PATH = Path(__file__).parent / "data" / "cookie_cats.csv"


@st.cache_data
def load_cookie_cats() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def render_ab_result(label_a, label_b, sa, na, sb, nb, alpha, alternative):
    """Run the two-proportion test and render metrics, verdict, CI, and a chart.

    Convention: A is control, B is treatment, and a HIGHER rate is better.
    """
    result = two_proportion_ztest(sa, na, sb, nb, alpha=alpha, alternative=alternative)
    conf = int(round((1 - alpha) * 100))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{label_a} (control)", f"{result.rate_a:.2%}", help=f"{sa:,} / {na:,}")
    c2.metric(f"{label_b} (treatment)", f"{result.rate_b:.2%}", help=f"{sb:,} / {nb:,}")
    c3.metric("Abs. difference (B−A)", f"{result.abs_diff:+.2%}")
    c4.metric("Relative lift", f"{result.rel_lift:+.1%}")

    if not result.significant:
        st.warning(
            f"**No significant difference** (p = {result.p_value:.4f} at α = {alpha}). "
            "Don't ship on this evidence."
        )
    elif result.abs_diff > 0:
        st.success(f"**Ship** — treatment beats control by {result.abs_diff:+.2%} (p = {result.p_value:.4f}).")
    else:
        st.error(
            f"**Don't ship** — treatment is worse than control by {result.abs_diff:.2%} "
            f"(p = {result.p_value:.4f})."
        )

    st.markdown(
        f"- **z = {result.z_stat:.3f}** &nbsp; **p = {result.p_value:.5f}** &nbsp; (_{alternative}_)\n"
        f"- **{conf}% CI on the difference (B − A):** [{result.ci_low:+.2%}, {result.ci_high:+.2%}]"
    )

    ci_a = proportion_ci(sa, na, alpha)
    ci_b = proportion_ci(sb, nb, alpha)
    fig = go.Figure()
    fig.add_bar(
        x=[str(label_a), str(label_b)],
        y=[result.rate_a, result.rate_b],
        marker_color=["#8b949e", "#5b8cff"],
        error_y=dict(
            type="data",
            symmetric=False,
            array=[ci_a[1] - result.rate_a, ci_b[1] - result.rate_b],
            arrayminus=[result.rate_a - ci_a[0], result.rate_b - ci_b[0]],
        ),
    )
    fig.update_layout(
        title=f"Conversion rate by variant ({conf}% CI)",
        yaxis_title="Rate",
        yaxis_tickformat=".1%",
        template="plotly_white",
        height=360,
        margin=dict(l=10, r=10, t=50, b=10),
    )
    st.plotly_chart(fig, width="stretch")

    with st.expander("Test & assumptions"):
        st.markdown(
            "- **Test:** pooled two-proportion *z*-test (variance under H0 uses the pooled rate). "
            "The CI on the difference uses the unpooled SE, so significance and the CI agree.\n"
            "- **Valid when:** two independent groups, binary outcome, a **single read** at a pre-set "
            "sample size (no peeking), and `n·p`, `n·(1−p)` ≳ 10 per group.\n"
            "- The ship/no-ship call assumes a **higher rate is better** and treats **B as the treatment**."
        )


design_tab, analyze_tab, pitfalls_tab = st.tabs(["Design", "Analyze", "Pitfalls"])


# --------------------------------------------------------------------------- #
# Design tab — sample-size calculator + sensitivity curve  (LIVE)
# --------------------------------------------------------------------------- #
with design_tab:
    st.subheader("Sample-size designer")
    st.write(
        "Plan a fixed-horizon two-proportion test: choose your baseline rate and the "
        "smallest lift worth detecting, and see how many users each variant needs."
    )

    left, right = st.columns([1, 1.4], gap="large")

    with left:
        baseline_pct = st.slider(
            "Baseline conversion rate", 0.5, 50.0, 10.0, step=0.5, format="%.1f%%",
            help="Control group's current conversion rate.",
        )
        baseline = baseline_pct / 100.0

        mde_pct = st.slider(
            "Minimum detectable effect (absolute)", 0.1, 10.0, 2.0, step=0.1, format="%.1f pp",
            help="Smallest ABSOLUTE change in rate you want to reliably detect, in percentage points. "
                 "2.0 pp means detecting a move from 10% to 12%.",
        )
        mde = mde_pct / 100.0

        power = st.slider("Statistical power (1 − β)", 0.50, 0.99, 0.80, step=0.01)
        alpha = st.select_slider(
            "Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05
        )
        alternative = st.radio(
            "Alternative hypothesis",
            options=["two-sided", "larger", "smaller"],
            horizontal=True,
            help="Two-sided unless you only care about an improvement (larger) or a regression (smaller).",
        )

    # Guard: treatment rate must stay below 100%.
    if baseline + mde >= 1.0:
        right.error("Baseline + MDE must be below 100%. Lower the baseline or the MDE.")
    else:
        n_per_group = required_sample_size(
            baseline, mde, alpha=alpha, power=power, alternative=alternative
        )
        achieved = power_for_sample_size(baseline, mde, n_per_group, alpha=alpha, alternative=alternative)

        with right:
            m1, m2, m3 = st.columns(3)
            m1.metric("Per variant", f"{n_per_group:,}")
            m2.metric("Total (2 variants)", f"{2 * n_per_group:,}")
            m3.metric("Achieved power", f"{achieved:.0%}")

            st.markdown(
                f"To detect a change from **{baseline:.1%}** to **{baseline + mde:.1%}** "
                f"at **α = {alpha}** with **{power:.0%}** power, you need "
                f"**{n_per_group:,} users per variant** ({2 * n_per_group:,} total)."
            )

            # Sensitivity curve: required n across a range of MDEs.
            mde_lo = max(0.0025, mde / 4)
            mde_hi = min(0.30, mde * 2)
            grid, sizes = sensitivity_curve(
                baseline, mde_lo, mde_hi, alpha=alpha, power=power, alternative=alternative
            )
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=grid * 100, y=sizes, mode="lines", name="Required n / variant",
                    line=dict(width=3),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=[mde * 100], y=[n_per_group], mode="markers",
                    name="Your design", marker=dict(size=12, symbol="diamond"),
                )
            )
            fig.update_layout(
                title="Sample size vs. minimum detectable effect",
                xaxis_title="MDE (percentage points)",
                yaxis_title="Required users per variant",
                template="plotly_white",
                height=380,
                margin=dict(l=10, r=10, t=50, b=10),
            )
            st.plotly_chart(fig, width="stretch")

    with st.expander("Test & assumptions"):
        st.markdown(
            "- **Test:** two-proportion comparison, normal approximation (Cohen's *h* effect size via "
            "statsmodels `NormalIndPower`).\n"
            "- **Valid when:** two independent groups, binary outcome, a **single read at the planned "
            "sample size** (no peeking), and `n·p`, `n·(1−p)` ≳ 10 per group.\n"
            "- **Equal allocation** (1:1) is assumed here."
        )


# --------------------------------------------------------------------------- #
# Analyze tab — roadmap
# --------------------------------------------------------------------------- #
with analyze_tab:
    st.subheader("Analyze a result")

    opt1, opt2 = st.columns([1.5, 1], gap="large")
    source = opt1.radio(
        "Data source",
        ["Cookie Cats (bundled)", "Upload CSV", "Enter summary counts"],
        horizontal=True,
        key="an_source",
    )
    alpha = opt2.select_slider("Significance level (α)", options=[0.01, 0.05, 0.10], value=0.05, key="an_alpha")
    alternative = opt2.radio(
        "Alternative", ["two-sided", "larger", "smaller"], horizontal=True, key="an_alt"
    )
    st.divider()

    if source == "Cookie Cats (bundled)":
        if DATA_PATH.exists():
            df = load_cookie_cats()
            st.caption(
                f"**Cookie Cats** — {len(df):,} players. Control **gate_30** vs treatment **gate_40** "
                "(the first in-game gate moved from level 30 → 40). A real 2016 mobile-game experiment."
            )
            metric = st.radio(
                "Retention metric",
                ["retention_1", "retention_7"],
                horizontal=True,
                help="retention_1 = returned 1 day after install; retention_7 = returned 7 days after install.",
                key="an_metric",
            )
            s = summarize_binary_experiment(df, "version", metric, "gate_30", "gate_40")
            render_ab_result(
                "gate_30", "gate_40",
                s.successes_control, s.n_control,
                s.successes_treatment, s.n_treatment,
                alpha, alternative,
            )
        else:
            st.info("Bundled `data/cookie_cats.csv` not found — see `data/README.md` for the one-line download.")

    elif source == "Upload CSV":
        up = st.file_uploader("Tidy CSV — one row per unit", type=["csv"])
        if up is None:
            st.caption("Your CSV needs a **group** column and a **binary outcome** column (0/1 or True/False).")
        else:
            df = pd.read_csv(up)
            st.dataframe(df.head(), width="stretch")
            cols = list(df.columns)
            m1, m2 = st.columns(2)
            group_col = m1.selectbox("Group column", cols)
            outcome_col = m2.selectbox("Binary outcome column", cols)
            groups = available_groups(df, group_col)
            if len(groups) < 2:
                st.warning("The group column needs at least two distinct values.")
            else:
                g1, g2 = st.columns(2)
                control = g1.selectbox("Control (A)", groups, index=0)
                treatment = g2.selectbox("Treatment (B)", groups, index=min(1, len(groups) - 1))
                try:
                    s = summarize_binary_experiment(df, group_col, outcome_col, control, treatment)
                    render_ab_result(
                        control, treatment,
                        s.successes_control, s.n_control,
                        s.successes_treatment, s.n_treatment,
                        alpha, alternative,
                    )
                except ValueError as exc:
                    st.error(str(exc))

    else:  # Enter summary counts
        st.caption("Already have the totals? Enter conversions and sample sizes directly.")
        m1, m2 = st.columns(2)
        sa = m1.number_input("Control conversions", min_value=0, value=200, step=1)
        na = m1.number_input("Control sample size", min_value=1, value=1000, step=1)
        sb = m2.number_input("Treatment conversions", min_value=0, value=240, step=1)
        nb = m2.number_input("Treatment sample size", min_value=1, value=1000, step=1)
        if sa > na or sb > nb:
            st.warning("Conversions cannot exceed sample size.")
        else:
            render_ab_result("Control", "Treatment", int(sa), int(na), int(sb), int(nb), alpha, alternative)


# --------------------------------------------------------------------------- #
# Pitfalls tab — roadmap
# --------------------------------------------------------------------------- #
with pitfalls_tab:
    st.subheader("Pitfalls")
    st.info(
        "🚧 **Coming next.** An interactive **peeking simulator** (watch the false-positive rate inflate "
        "when you check results early) and a **CUPED** variance-reduction demo."
    )
    st.caption(
        "CUPED is already implemented and tested in `stats/cuped.py` (`cuped_adjust`); "
        "only this UI is pending."
    )


st.divider()
st.caption(
    "Built with Streamlit · statsmodels · SciPy · Plotly. "
    "Educational tool — verify against your own analysis before making ship decisions."
)
