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
- **UTC time format while reasembling to hours** —  changed time format to UTC while resampling dataset to hours to avoid mistakes with two same existing hours (and moving alert one hour back/forward). After resampling dataset — changing back to local TZ. 
- **Ignoring last and first alert** — If the last alarm is long and ends within the next hour (finished_at is more than started_at.max()), the last hour or two of its tail may not hit the dataset, but that doesn't affect the model. Was fixed through changing strted_at.max() to finished_at.max().
**Rejected** — it took false data from September 2025 that was cuted of. Changed to end=series_end.
- **Structural break** — alert system differentiation (2025). During EDA, the monthly alert-rate plot revealed an anomalous band of near-100% alert rate and a distorted signal from 2024–2025. Investigation (confirmed via news sources) traced this to a change in how alerts are recorded, not a change in actual strike activity — Kharkiv oblast led Ukraine in alert frequency throughout 2025. Kharkiv introduced a differentiated alert system in two stages: the city was separated from the oblast (22 Feb 2025), and the oblast was subdivided to raion/hromada level (1 Aug 2025). Threats previously logged as a single oblast-wide alert became multiple independent sub-regional records.  
**This is a structural break, not concept drift:** the unit of observation changed. Pre-reform, the target was a single oblast-wide siren; post-reform, it is the union of ~7 independent raion sirens.  
**Hypothesis tested and rejected:** I first attempted to reconstruct an oblast-wide target by aggregating alerts across all administrative levels (oblast/raion/hromada). EDA showed this produces a degenerate target — with multiple raions holding independent alert windows, their union covers nearly all post-reform hours (alert rate → ~1.0), so the signal would be almost constant and unpredictable in a meaningful sense. The aggregated signal is not equivalent to the pre-reform oblast signal.  
**Resolution:** I restricted the dataset to level == "oblast" records and capped the series at the reform point (~August 2025), yielding a temporally homogeneous target from March 2022 to August 2025 (~3.5 years, ~29k hourly rows) where "oblast under alert" carries a single consistent meaning throughout. This is a deliberate design choice driven by the recording-system change, not a data-availability limitation — stitching across the discontinuity would require an arbitrary cutoff and introduce new inconsistency, which is not justified for this scope.  
**Data-quality note:** two corrupt records were identified during this investigation that alone spanned every hour from June 2024 to January 2026 (the spurious 1.0 band); these were removed.
**EDA result** — located into the **Exploratory Data Analysis** section

## Data source
- **Vadimkin/ukrainian-air-raid-sirens-dataset** — Flat CSV (no api code, just cloning through git); data starting from 2022 (37k rows for Kharkiv); updated daily; MIT license — free to use. Use official_data_en.csv from git repository.

## Exploratory Data Analysis
After resolving the structural break, the series spans March 2022 – August 2025
(~29k hourly rows) with a **base rate of 0.48** — the oblast was under alert in
roughly half of all hours. The target is therefore close to balanced, so accuracy
is not catastrophically misleading here; nonetheless PR-AUC, F1 and recall are
reported as primary metrics, since the operational cost of a missed alert
outweighs that of a false alarm.
### Monthly alert rate
![Monthly alert rate](figures/01_monthly_alert_rate.png)
The artificial collapse is gone and the series is temporally homogeneous. A genuine
upward drift in alert intensity is visible through 2024–2025, reflecting real
changes in strike activity (concept drift) — which motivates the strictly temporal
train/test split.
### Seasonality: hour-of-day × weekday
![Hour × weekday heatmap](figures/02_seasonality_heatmap.png)
Alert probability is elevated in late-evening/night hours (≈22:00–01:00) and lowest
in early morning (≈06:00–08:00), consistently across the week. A weak day-of-week
effect is visible (mid-week nights slightly higher, weekends slightly lower), but it
is a secondary signal relative to time-of-day.
### STL decomposition
![STL decomposition](figures/03_stl_decomposition.png)
The trend component dominates (range ≈0.2–1.0), the weekly seasonal component is weak
(amplitude ≈±0.1), and the residual is large with pronounced spikes during
intensive-strike periods. This confirms that alert dynamics are driven mainly by slow
shifts in war intensity and exogenous strike events rather than smooth calendar
cycles — bounding achievable predictability and framing the task as
conditional-probability estimation rather than deterministic forecasting.
### Autocorrelation (ACF / PACF)
![ACF and PACF](figures/04_acf_pacf.png)
Strong autocorrelation at lag 1 (≈0.50) with a sharp PACF cutoff after lag 1 indicates
an **AR(1)-like process** — the immediately preceding hour carries most of the
predictive signal. A long, low ACF tail (≈0.15 out to 48h) reflects mild
persistence/clustering, while the absence of a strong 24h peak shows daily periodicity
is weak.
**Implication for modeling:** the previous-hour state (`alert_{t-1}`) is expected to be the dominant feature; recent-activity lags and rolling counts add secondary signal; and a persistence baseline will be a strong competitor that the models must beat. EDA thus predicts — before any model is trained — that gains over persistence will be bounded by the exogenous nature of strikes. This prediction is revisited in the results section.
## (to fill) Features & leakage prevention
## (to fill) Models & results
## (to fill) Limitations