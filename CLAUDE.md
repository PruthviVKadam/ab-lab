# Experimentation Lab — CLAUDE.md
Streamlit app for designing & analyzing A/B tests. Tabs: Design / Analyze / Pitfalls. Python 3.11+.

## Commands
- Setup: pip install -r requirements-dev.txt
- Run: streamlit run app.py
- Test: python -m pytest (all stats functions have tests against known textbook / reference values)
- Deploy: push to main → Streamlit Community Cloud auto-redeploys

## Structure
- app.py — UI only, no statistics inline (a thin render helper that calls pure stats is OK)
- stats/power.py · stats/tests.py · stats/aggregate.py · stats/cuped.py — pure functions, fully unit-tested
- tests/ — pytest suite (cross-checked vs statsmodels / hand-computed oracles)
- data/cookie_cats.csv — BUNDLED (90,189 rows; committed with `git add -f`; never modify). See data/README.md.

## Status
- Design tab: LIVE (sample-size calculator + sensitivity curve).
- Analyze tab: LIVE (Cookie Cats bundled / CSV upload / manual counts → two-proportion z-test, CI, verdict).
- Pitfalls tab: UI roadmap; CUPED stats (cuped_adjust) already implemented and tested.

## Rules
- Every test function docstring states H0, H1, assumptions, and when the test is INVALID.
- UI must display which test was used + its assumptions next to every p-value / sample-size output.
- Verify outputs against statsmodels/scipy reference implementations before claiming correctness.
- No metrics in README except ones copied from actual app/test output. No datasets >10MB in git.
- Do not claim a tab/feature exists in the README until its UI is actually shipped.
