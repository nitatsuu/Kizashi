# Kizashi — Air Raid Alert Forecasting (Kharkiv)

## Problem framing
Binary hourly forecast: will there be an air raid alert in Kharkiv
in the next hour.

## Key decisions
- **Binary target (alert yes/no per hour)** — chose occurrence over
duration/intensity to keep the target clean and the problem tractable in the 2-day scope.
- **Region: Kharkiv** — front-line, so alerts occur not only on mass-strike
days; richer signal than a deep-rear region.
- **No database** — ~37k hourly rows; flat file + pandas is sufficient,
a DB would be over-engineering.
- **Data: 2022–present, not truncated** — longer history captures annual seasonality; possible concept drift since 2022 addressed via temporal split.
- **Imputed alert ends (naive=True) retained** — target is occurrence, not duration; imputation noise affects only duration. Share of naive: 4 out of 13308. Dataset has no native "naive=True" feature (only wrote about it in README they haven't implemented it) though all air raids that are equal to 30 minutes are counted as naive.
- **Source=official** — original repository has two sources; decided to use official one.
- **TZ=Europe/Kyiv** — using timezone instead of hardcore (UTC+3) to prevent problem with Winter\Summer time.
- **UTC time format while reasembling to hours** —  changed time format to UTC while reasembling dataset to hours to avoid mistakes with two same existing hours (and moving alert one hour back/forward). After reasembling dataset — changing back to local TZ. 
- **Ignoring last and first alert** — If the last alarm is long and ends within the next hour (finished_at is more than started_at.max()), the last hour or two of its tail may not hit the dataset, but that doesn't affect the model. Was fixed through changing strted_at.max() to finished_at.max().

## Data source
- **Vadimkin/ukrainian-air-raid-sirens-dataset** — Flat CSV (no api code, just cloning through git); data starting from 2022 (37k rows for Kharkiv); updated daily; MIT license — free to use.

## (to fill) Features & leakage prevention
## (to fill) Models & results
## (to fill) Limitations