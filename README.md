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
- **Imputed alert ends (naive=True) retained** — target is occurrence, not duration; imputation noise affects only duration. Share of naive: X%. Decision: add end to alert in 30 minutes.
- **Source=official** — original repository has two sources; decided to use official one.

## Data source
- **Vadimkin/ukrainian-air-raid-sirens-dataset** — Flat CSV (no api code, just cloning through git); data starting from 2022 (37k rows for Kharkiv); updated daily; MIT license — free to use.

## (to fill) Features & leakage prevention
## (to fill) Models & results
## (to fill) Limitations