"""Experimentation Lab — A/B Test Designer & Analyzer.

UI only. All statistics live in the `stats/` package (pure, unit-tested).
Tabs: Design (live) · Analyze (roadmap) · Pitfalls (roadmap).
"""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from stats.power import power_for_sample_size, required_sample_size, sensitivity_curve

st.set_page_config(page_title="Experimentation Lab", page_icon="🧪", layout="wide")

st.title("🧪 Experimentation Lab")
st.caption("Design and analyze A/B tests — power analysis, peeking pitfalls, and variance reduction.")

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
            st.plotly_chart(fig, use_container_width=True)

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
    st.info(
        "🚧 **Coming next.** This tab will load the bundled Cookie Cats experiment (90K real users, "
        "gate_30 vs gate_40 retention) or your uploaded CSV, run the two-proportion z-test, and report "
        "the p-value, confidence interval, lift, and a plain-English ship / don't-ship verdict."
    )
    st.caption(
        "The statistics are already implemented and tested in `stats/tests.py` "
        "(`two_proportion_ztest`); only this UI is pending."
    )


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
