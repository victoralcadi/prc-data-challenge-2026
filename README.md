# PRC Data Challenge 2026 — Taxi-Out Time Prediction

Working repo for the [PRC Data Challenge 2026](https://prc-data-challenge-2026.netlify.app/):
predict **taxi-out time** (`TAXITIME_SEC_mvt`) for departures at 10 major European
airports.

- Target: `TAXITIME_SEC_mvt = MVT_TIME_UTC_mvt - BLOCK_TIME_UTC_mvt` for rows with `PHASE_mvt = "DEP"`
- Metric: RMSE (seconds), best submission per team counts
- Train: monthly parquet files covering all of 2025 — a published 4,167,797 movements
  (arrivals + departures) at 10 reporting airports: EDDF, EDDM, EGLL, EHAM, LEBL, LEMD,
  LFPG, LIRF, LTFM, LSZH. Roughly half are departures, so expect ~2.1M labelled targets.
  `prc2026 status --count` checks the parquet row counts against that figure.
- Rank: `ranking.parquet` (Jan + Jul 2026) with `BLOCK_TIME_UTC_mvt` and `TAXITIME_SEC_mvt`
  blanked for departures
- Submit: `submitting.parquet` template (`MVT_ID_mvt`, `TAXITIME_SEC_mvt`), uploaded as
  `<team-name>_v<n>.parquet`, max 5/day

Why it matters (per the [rationale](https://prc-data-challenge-2026.netlify.app/rationale.html)):
taxi-out is hard to measure and hard to predict, and a good estimator lets you quantify
excess fuel burn / CO2 from constrained ground operations in post-ops analysis.

## The modelling constraint that shapes everything

For the departures you must predict, **off-block time is not given**. Only the
*takeoff* time (`MVT_TIME_UTC_mvt`) and the *scheduled* time are known. So any feature
must be computable from:

- takeoff time (DEP) — known
- scheduled time — known
- runway, stand, aircraft type, operator, market segment — known
- arrival movements' block + movement times — known (only departures were blanked)

Congestion features here are therefore anchored on **takeoff time**, not push-back time.

> `AOBT_3_flt` (NM actual off-block) is a near-duplicate of the blanked
> `BLOCK_TIME_UTC_mvt`. `prc2026` keeps it out of the default feature set; run
> `prc2026 audit` to check whether it is populated in `ranking.parquet` before
> deciding what to do with it.

## Layout

```
src/prc2026/
  config.py     paths, column names, expected file list, env-driven settings
  manifest.py   the 14 objects the bucket should hold; local vs remote diff
  io_s3.py      MinIO/S3 access to the OSN buckets (list/download/upload)
  data.py       parquet loading, dtype normalisation, train/rank splits
  features.py   time, congestion/queue, and group-statistic features
  model.py      LightGBM training, month-holdout validation, prediction
  cli.py        `prc2026 download|status|audit|train|predict|upload`
scripts/quicklook.py   one-screen sanity check of a raw parquet file
data/raw/              downloaded parquet (git-ignored)
models/                trained artefacts (git-ignored)
submissions/           generated parquet submissions (git-ignored)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env    # then fill in your team credentials
```

### Credentials

Registration gives you a team name, team id, `bucket_access_key` and
`bucket_access_secret`; the keys are generated from your OpenSky Network account
(Account → Access Keys). Put them in `.env`:

```
PRC_ACCESS_KEY=...
PRC_SECRET_KEY=...
PRC_TEAM_NAME=team_something
PRC_SUBMISSION_BUCKET=team_something
```

### Get the data

Expected bucket layout — 14 parquet objects, no CSVs:

```
competition-data/
├── training_2025-01-01_2025-02-01.parquet   (21M)
├── training_2025-02-01_2025-03-01.parquet   (19M)
├── training_2025-03-01_2025-04-01.parquet   (22M)
├── training_2025-04-01_2025-05-01.parquet   (23M)
├── training_2025-05-01_2025-06-01.parquet   (25M)
├── training_2025-06-01_2025-07-01.parquet   (24M)
├── training_2025-07-01_2025-08-01.parquet   (25M)
├── training_2025-08-01_2025-09-01.parquet   (25M)
├── training_2025-09-01_2025-10-01.parquet   (24M)
├── training_2025-10-01_2025-11-01.parquet   (25M)
├── training_2025-11-01_2025-12-01.parquet   (22M)
├── training_2025-12-01_2026-01-01.parquet   (22M)
├── ranking.parquet     (27M)  Jan + Jul 2026, departure block/taxi times blanked
└── submitting.parquet  (1.1M) template: MVT_ID_mvt + empty TAXITIME_SEC_mvt
```

Each file spans `[start, end)`, so `training_2025-12-01_2026-01-01.parquet` is December.
`manifest.py` holds this list; `prc2026 status` diffs it against what you actually have.

> Not to be confused with the **2024** challenge bucket, which held daily trajectory
> files (`2022-01-01.parquet`, …) plus `challenge_set.csv`, `submission_set.csv` and
> `final_submission_set.csv`. That challenge predicted take-off weight; only its MinIO
> setup instructions carry over.

Python (no extra tooling):

```bash
prc2026 download --list                  # show buckets/objects you can see
prc2026 download                         # pull competition data into data/raw/
prc2026 status --remote                  # expected vs local vs bucket
```

Or with the [MinIO client](https://min.io/docs/minio/linux/reference/minio-mc.html):

```bash
mc alias set dc26 https://s3.opensky-network.org/ "$PRC_ACCESS_KEY" "$PRC_SECRET_KEY"
mc ls dc26
mc cp --recursive dc26/competition-data/ data/raw/
```

## Workflow

```bash
prc2026 status                # are all 14 files there, at roughly the right size?
prc2026 audit                 # column availability + target sanity, train vs ranking
prc2026 train                 # month-holdout validation (Jan + Jul), then full refit
prc2026 predict --version 1   # writes submissions/<team>_v1.parquet
prc2026 upload --version 1    # copies it to your submission bucket
```

`train` prints validation RMSE in seconds. The naive baseline (per airport median) is
reported alongside so you can tell whether a change actually bought anything.

## Notes

- The data page opens by saying **11** airports and then states the 4,167,797 total is
  across the **10** airports in its Table 1, which does list 10. The code follows the
  table. If an eleventh airport shows up in the data, `airport` in `AIRPORT_TZ` will miss
  it and local hour silently falls back to UTC — `prc2026 audit` will surface the unknown
  code first.
- The data is real and messy: movement and flight records do not always agree, and the
  organisers deliberately left the inconsistencies in place.
- Military, Head of State and sensitive movements have been removed.
- Times are UTC. Local time-of-day still matters operationally, so airport-local hour is
  derived in `features.py`.
