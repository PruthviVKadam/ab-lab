# 🧪 Experimentation Lab — A/B Test Designer & Analyzer

An interactive Streamlit app for **designing and reading A/B tests correctly** — power
analysis, the peeking trap, and variance reduction (CUPED). Built to turn experimentation
theory into something you can drive live.

**Live demo:** [ab-lab-test.streamlit.app](https://ab-lab-test.streamlit.app/)

---

## Problem → Approach → Result

- **Problem:** A/B tests are easy to get wrong — underpowered designs, peeking at results
  early, and ignoring variance reduction all lead to bad ship decisions.
- **Approach:** A three-tab tool. **Design** computes the sample size you need (and shows the
  cost of detecting smaller effects); **Analyze** runs a two-proportion z-test on a real 90K-user
  experiment, an uploaded CSV, or manual counts; **Pitfalls** simulates how peeking inflates false
  positives and how CUPED cuts variance.
- **Result:** Every statistic is a pure, unit-tested function cross-checked against statsmodels
  and hand-computed values — so the numbers on screen are trustworthy.

## Insights

- **A "significant" result can argue *against* shipping.** Moving the gate to level 40 produced a
  statistically significant **drop** in 7-day retention (18.20% vs 19.02%, p = 0.00155) — the right
  call was *don't ship*. Significance is direction-agnostic; the business reading is everything.
- **Peeking is the silent killer.** Checking a test repeatedly and stopping at the first p < 0.05
  turns a 5% false-positive rate into **~24% at 20 looks** — many "wins" from unmonitored dashboards
  are noise.
- **Variance reduction beats a bigger sample.** A pre-experiment covariate (CUPED) cuts the standard
  error by ~√(1−ρ²) at zero bias — ρ = 0.9 gives a ~56% smaller SE, i.e. the same power for a
  fraction of the traffic.

## Current status (honest)

| Tab | State |
| --- | --- |
| **Design** — sample-size calculator + sensitivity curve | ✅ Live |
| **Analyze** — two-proportion z-test on Cookie Cats / CSV upload / manual counts | ✅ Live |
| **Pitfalls** — peeking/early-stopping simulator + CUPED demo | ✅ Live |

## What the Analyze tab shows (real Cookie Cats output)

The bundled [Cookie Cats](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats)
experiment (90,189 players) moved the first in-game gate from level 30 (control, `gate_30`) to
level 40 (treatment, `gate_40`). Running the app's two-proportion z-test — numbers copied
verbatim from the app output:

| Metric | gate_30 | gate_40 | Δ (B−A) | p-value | Verdict |
| --- | --- | --- | --- | --- | --- |
| `retention_1` | 44.82% | 44.23% | −0.59 pp | 0.07441 | No significant difference — don't ship |
| `retention_7` | 19.02% | 18.20% | −0.82 pp | 0.00155 | Significant **decrease** — don't ship |

Takeaway: moving the gate to level 40 **hurt** 7-day retention — a clean example of a
statistically significant result that argues *against* shipping.

## What the Pitfalls tab shows

Both demos are Monte-Carlo (seeded, reproducible). Example output at α = 0.05:

- **Peeking:** under the null (no real effect), testing once gives a **5.0%** false-positive rate,
  but stopping early whenever p < α inflates it to **~18.7% at 10 looks** and **~24.4% at 20 looks**.
- **CUPED:** a pre-experiment covariate of correlation ρ shrinks the effect estimate's standard
  error by ~√(1 − ρ²) without bias — e.g. **ρ = 0.6 → ~20% smaller SE**, **ρ = 0.9 → ~56% smaller**.

## Run locally

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
streamlit run app.py
```

## Test

```bash
python -m pytest
```

The suite (59 tests) validates power analysis against an independent textbook sample-size
formula, the two-proportion z-test against `statsmodels.proportions_ztest` and a hand-computed
example, the aggregation helper against the bundled Cookie Cats counts, CUPED against the
`1 − corr²` variance-reduction identity, and the peeking simulator's false-positive rate against
its nominal α. A headless `streamlit.testing` smoke test exercises every tab and data source.

## Project layout

```text
app.py                # UI only — no statistics inline
stats/power.py        # sample size, achieved power, sensitivity curve
stats/tests.py        # two-proportion z-test (pooled), per-group CI, plain-English verdict
stats/aggregate.py    # tidy dataframe -> per-variant success/trial counts
stats/cuped.py        # CUPED variance reduction + simulated demo
stats/peeking.py      # Monte-Carlo peeking / early-stopping false-positive simulator
tests/                # pytest suite (textbook / reference cross-checks)
data/cookie_cats.csv  # bundled experiment (90,189 rows)
```

## Stack

Python · Streamlit · statsmodels · SciPy · pandas · Plotly. Deployed free on Streamlit Community Cloud.

## The stats, briefly

- **Sample size** uses Cohen's *h* with statsmodels `NormalIndPower` (two-sided by default).
- **Analysis** uses a pooled-variance two-proportion z-test; the reported CI on the absolute
  difference uses the unpooled SE, so significance and the CI agree.
- **CUPED** subtracts `θ·(x − x̄)` where `θ = cov(y, x)/var(x)`, preserving the mean while reducing
  variance by `corr(x, y)²`.

---

*Educational tool. Validate against your own analysis before making real ship decisions.*
