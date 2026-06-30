"""Experimentation Lab — A/B Test Designer & Analyzer.

UI only. All statistics live in the `stats/` package (pure, unit-tested).
Tabs: Design (live) · Analyze (live) · Pitfalls (live).
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from stats.aggregate import available_groups, summarize_binary_experiment
from stats.cuped import cuped_demo
from stats.peeking import peeking_fpr_curve, run_peeking_simulation
from stats.power import power_for_sample_size, required_sample_size, sensitivity_curve
from brand import apply_brand
from stats.tests import proportion_ci, two_proportion_ztest

st.set_page_config(page_title="Experimentation Lab", page_icon="🧪", layout="wide")
apply_brand()

st.title("🧪 Experimentation Lab")
st.caption("Design and analyze A/B tests — power analysis, peeking pitfalls, and variance reduction.")

DATA_PATH = Path(__file__).parent / "data" / "cookie_cats.csv"


@st.cache_data
def load_cookie_cats() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def cached_peeking(true_rate, n_per_group, n_looks, alpha, n_sims):
    return run_peeking_simulation(true_rate, n_per_group, n_looks, alpha, n_sims, seed=0)


@st.cache_data
def cached_fpr_curve(true_rate, n_per_group, max_looks, alpha, n_sims):
    return peeking_fpr_curve(true_rate, n_per_group, max_looks, alpha, n_sims, seed=0)


@st.cache_data
def cached_cuped_demo(rho, n_users, true_effect):
    return cuped_demo(rho, n_users, true_effect, seed=0)


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
# Pitfalls tab — peeking simulator + CUPED demo  (LIVE)
# --------------------------------------------------------------------------- #
with pitfalls_tab:
    st.subheader("Pitfalls")
    demo = st.radio(
        "Demo", ["Peeking / early stopping", "CUPED variance reduction"], horizontal=True, key="pit_demo"
    )
    st.divider()

    if demo == "Peeking / early stopping":
        st.markdown(
            "Under the **null hypothesis** (A and B are truly equal), testing once gives a false-positive "
            "rate ≈ α. But if you check repeatedly and stop the moment p < α, the chance of a false 'win' "
            "balloons. Both groups below share the **same true rate** — every rejection is a fluke."
        )
        c1, c2, c3 = st.columns(3)
        pk_rate = c1.slider("True rate (both groups)", 0.05, 0.50, 0.20, 0.01, key="pk_rate")
        pk_n = c2.slider("Users per group", 500, 4000, 2000, 250, key="pk_n")
        pk_looks = c3.slider("Number of looks (peeks)", 1, 20, 10, 1, key="pk_looks")
        pk_alpha = c1.select_slider("Significance level (α)", [0.01, 0.05, 0.10], 0.05, key="pk_alpha")
        pk_sims = c2.select_slider("Simulated experiments", [1000, 2000, 4000], 2000, key="pk_sims")

        res = cached_peeking(pk_rate, pk_n, pk_looks, pk_alpha, pk_sims)
        looks_axis, fprs = cached_fpr_curve(pk_rate, pk_n, 20, pk_alpha, pk_sims)

        k1, k2, k3 = st.columns(3)
        k1.metric("Nominal α", f"{pk_alpha:.0%}")
        k2.metric("False-positive rate — single look", f"{res.fpr_single:.1%}")
        k3.metric(
            f"False-positive rate — peeking {pk_looks}×",
            f"{res.fpr_peeking:.1%}",
            delta=f"{res.fpr_peeking - pk_alpha:+.1%} vs α",
            delta_color="inverse",
        )

        fig = go.Figure()
        fig.add_scatter(x=looks_axis, y=fprs, mode="lines+markers", name="Peeking FPR", line=dict(width=3))
        fig.add_scatter(
            x=[pk_looks], y=[res.fpr_peeking], mode="markers",
            marker=dict(size=13, symbol="diamond", color="#d62728"), name="Your setting",
        )
        fig.add_hline(y=pk_alpha, line_dash="dash", line_color="#8b949e",
                      annotation_text=f"nominal α = {pk_alpha}")
        fig.update_layout(
            title="False-positive rate vs. number of looks",
            xaxis_title="number of looks (peeks)",
            yaxis_title="P(at least one false 'win')",
            yaxis_tickformat=".0%", template="plotly_white", height=330,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        fig2 = go.Figure()
        n_ex = res.example_pvalues.shape[0]
        for i in range(n_ex):
            crossed = bool((res.example_pvalues[i] < pk_alpha).any())
            fig2.add_scatter(
                x=res.look_sizes, y=res.example_pvalues[i], mode="lines",
                line=dict(width=1, color="rgba(214,39,39,0.55)" if crossed else "rgba(139,148,158,0.35)"),
                showlegend=False, hoverinfo="skip",
            )
        fig2.add_hline(y=pk_alpha, line_dash="dash", line_color="#1b1f29",
                       annotation_text=f"α = {pk_alpha}")
        fig2.update_layout(
            title=f"Running p-value of {n_ex} null experiments (red = crossed below α at some look)",
            xaxis_title="cumulative users per group", yaxis_title="p-value",
            template="plotly_white", height=330, margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig2, width="stretch")

        crossed_frac = float((res.example_pvalues < pk_alpha).any(axis=1).mean())
        st.caption(
            f"There is **no real difference**, yet {crossed_frac:.0%} of these example experiments dipped "
            "below α at some point. Fix your sample size in advance, or use sequential-testing corrections."
        )

    else:  # CUPED
        st.markdown(
            "**CUPED** uses a pre-experiment covariate (e.g. each user's activity *before* the test) to "
            "strip predictable variance from the metric — shrinking the standard error and boosting power "
            "**without biasing** the effect estimate. The stronger the correlation, the bigger the win."
        )
        c1, c2, c3 = st.columns(3)
        cp_rho = c1.slider("Corr(pre-metric, outcome) ρ", 0.0, 0.95, 0.60, 0.05, key="cp_rho")
        cp_n = c2.slider("Users (total)", 1000, 40000, 10000, 1000, key="cp_n")
        cp_eff = c3.slider("True treatment effect Δ", 0.0, 0.50, 0.10, 0.01, key="cp_eff")

        d = cached_cuped_demo(cp_rho, cp_n, cp_eff)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Variance removed (ρ²)", f"{d.reduction:.0%}")
        k2.metric("SE — raw", f"{d.raw_se:.4f}")
        k3.metric("SE — CUPED", f"{d.cuped_se:.4f}", delta=f"{d.se_ratio - 1:+.0%}", delta_color="inverse")
        k4.metric("True effect Δ", f"{d.true_effect:.2f}")

        fig = go.Figure()
        fig.add_bar(
            x=["Raw metric", "CUPED-adjusted"],
            y=[d.raw_estimate, d.cuped_estimate],
            marker_color=["#8b949e", "#5b8cff"],
            error_y=dict(type="data", array=[1.96 * d.raw_se, 1.96 * d.cuped_se]),
        )
        fig.add_hline(y=d.true_effect, line_dash="dash", line_color="#3fb950",
                      annotation_text=f"true Δ = {d.true_effect:.2f}")
        fig.update_layout(
            title="Treatment-effect estimate ± 95% CI",
            yaxis_title="estimated effect", template="plotly_white", height=360,
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig, width="stretch")

        st.markdown(
            f"- Raw estimate **{d.raw_estimate:+.4f}** (95% CI ±{1.96 * d.raw_se:.4f}) · "
            f"CUPED estimate **{d.cuped_estimate:+.4f}** (95% CI ±{1.96 * d.cuped_se:.4f}).\n"
            f"- Both recover the true Δ = {d.true_effect:.2f}, but CUPED's interval is "
            f"**{(1 - d.se_ratio):.0%} narrower** — the same power for fewer users.\n"
            f"- Realized correlation CUPED exploited: ρ = {d.rho_actual:.2f} "
            f"(theoretical SE shrink ≈ √(1 − ρ²) = {(1 - d.reduction) ** 0.5:.2f})."
        )

    with st.expander("How these are computed"):
        st.markdown(
            "- **Peeking:** Monte-Carlo over many simulated *null* experiments; at each look a pooled "
            "two-proportion z-test is run on the data so far, and 'peeking' rejects if **any** look "
            "crosses α (`stats/peeking.py`).\n"
            "- **CUPED:** adjusts the metric to `y − θ·(x − x̄)` with `θ = cov(y, x)/var(x)`, preserving "
            "the mean while cutting variance by `corr(x, y)²` (`stats/cuped.py`).\n"
            "- Both modules are pure and unit-tested; the UI only renders their output."
        )


st.divider()
st.caption(
    "Built with Streamlit · statsmodels · SciPy · Plotly. "
    "Educational tool — verify against your own analysis before making ship decisions."
)
