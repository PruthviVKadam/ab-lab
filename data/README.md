# data/

## Cookie Cats A/B test (for the Analyze tab)
The Analyze tab will use the **Cookie Cats** mobile-game retention experiment:
~90,000 real users, randomized to `gate_30` vs `gate_40`, with 1-day and 7-day retention.

**Download:** Kaggle — search "Cookie Cats A/B testing" (e.g. dataset `yufengsui/mobile-games-ab-testing`).
Accept the dataset terms once, then place the CSV here as `cookie_cats.csv`.

This folder's CSVs are gitignored by default. When you're ready to bundle the dataset
with the app for deployment, add it explicitly:

```bash
git add -f data/cookie_cats.csv
```

Keep it under 10 MB (the raw Cookie Cats file is ~4 MB, well within limits).
**Never modify the raw file** — do any cleaning in code.
