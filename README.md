# 🧪 Experimentation Lab — A/B Test Designer & Analyzer

An interactive Streamlit app for **designing and reading A/B tests correctly** — power
analysis, the peeking trap, and variance reduction (CUPED). Built to turn experimentation
theory into something you can drive live.

**Live demo:** _deploying to Streamlit Community Cloud — link goes here once published._

---

## Problem → Approach → Result
- **Problem:** A/B tests are easy to get wrong — underpowered designs, peeking at results
  early, and ignoring variance reduction all lead to bad ship decisions.
- **Approach:** A three-tab tool. **Design** computes the sample size you need (and shows the
  cost of detecting smaller effects); **Analyze** runs a two-proportion z-test on a real 90K-user
  experiment; **Pitfalls** simulates how peeking inflates false positives and how CUPED helps.
- **Result:** Every statistic is a pure, unit-tested function cross-checked against statsmodels
  and hand-computed values — so the numbers on screen are trustworthy.

## Current status (honest)
| Tab | State |
|-----|-------|
| **Design** — sample-size calculator + sensitivity curve | ✅ Live |
| **Analyze** — two-proportion z-test on Cookie Cats / CSV upload | 🚧 Stats done & tested (`stats/tests.py`); UI pending |
| **Pitfalls** — peeking simulator + CUPED demo | 🚧 Stats done & tested (`stats/cuped.py`); UI pending |

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
The suite validates power analysis against an independent textbook sample-size formula, the
two-proportion z-test against `statsmodels.proportions_ztest` and a hand-computed example, and
CUPED against the `1 − corr²` variance-reduction identity.

## Project layout
```
app.py              # UI only — no statistics inline
stats/power.py      # sample size, achieved power, sensitivity curve
stats/tests.py      # two-proportion z-test (pooled), CI, plain-English verdict
stats/cuped.py      # CUPED variance reduction
tests/              # pytest suite (textbook / reference cross-checks)
data/               # Cookie Cats dataset goes here (see data/README.md)
```

## Stack
Python · Streamlit · statsmodels · SciPy · NumPy · Plotly. Deployed free on Streamlit Community Cloud.

## The stats, briefly
- **Sample size** uses Cohen's *h* with statsmodels `NormalIndPower` (two-sided by default).
- **Analysis** uses a pooled-variance two-proportion z-test; the reported CI on the absolute
  difference uses the unpooled SE, so significance and the CI agree.
- **CUPED** subtracts `θ·(x − x̄)` where `θ = cov(y, x)/var(x)`, preserving the mean while reducing
  variance by `corr(x, y)²`.

---
_Educational tool. Validate against your own analysis before making real ship decisions._
