# data/

## Cookie Cats A/B test (used by the Analyze tab)

`cookie_cats.csv` is **bundled in this repo** (committed with `git add -f`, ~2.6 MB) so the
deployed app works out of the box. It is the **Cookie Cats** mobile-game retention experiment:
90,189 real players randomized to `gate_30` vs `gate_40`, with 1-day and 7-day retention.

Columns: `userid`, `version` (`gate_30` | `gate_40`), `sum_gamerounds`, `retention_1` (bool),
`retention_7` (bool).

### Provenance

Public dataset, originally a DataCamp project, also on
[Kaggle](https://www.kaggle.com/datasets/mursideyarkin/mobile-games-ab-testing-cookie-cats).
This copy was fetched from a public GitHub mirror
([ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats](https://github.com/ryanschaub/Mobile-Games-A-B-Testing-with-Cookie-Cats))
and validated on download: shape `(90189, 5)`, expected columns, no nulls, no duplicate `userid`.

**Never modify the raw file** — do any cleaning in code.
