"""
Load raw alert CSV → filter Kharkiv oblast → build hourly binary series.
Output: data/kharkiv_hourly.parquet
"""
import pandas as pd

RAW_PATH = "data/official_data_en.csv"
OUT_PATH = "data/kharkiv_hourly.parquet"
TZ = "Europe/Kyiv"
# Aug 1 2025: Kharkiv switched from oblast-level to raion/hromada recording.
# Oblast records drop from ~190/month to 44 in Aug, then 0 in Sep.
# Cap here for a consistent target definition across the full series.
CUTOFF_UTC = "2025-08-01 00:00:00+00:00"


def load_kharkiv_alerts() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH, parse_dates=["started_at", "finished_at"])
    df = df[(df["oblast"] == "Kharkivska oblast") & (df["level"] == "oblast")].copy()
    # raw CSV contains exact duplicate rows — deduplicate before any processing
    df = df.drop_duplicates(subset=["oblast", "raion", "hromada", "level", "started_at", "finished_at"])
    df = df[df["started_at"] < CUTOFF_UTC]
    df["started_at"] = df["started_at"].dt.tz_convert(TZ)
    df["finished_at"] = df["finished_at"].dt.tz_convert(TZ)
    # alerts with duration == 30 min are likely imputed ends (naive)
    duration_min = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
    df["naive"] = (duration_min == 30).astype(int)
    return df[["started_at", "finished_at", "naive"]]


def expand_to_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Each alert row → one row per hour it touches (partial overlap counts).

    Arithmetic in UTC avoids DST fall-back ambiguity (e.g. 2024-10-27 03:00 Kyiv
    exists twice); tz_convert back to local only after all offsets are resolved.
    """
    records = []
    for _, row in df.iterrows():
        h = row["started_at"].tz_convert("UTC").floor("h")
        end = row["finished_at"].tz_convert("UTC")
        while h < end:
            records.append({"hour": h, "naive": row["naive"]})
            h += pd.Timedelta("1h")
    expanded = pd.DataFrame(records)
    if not expanded.empty:
        expanded["hour"] = expanded["hour"].dt.tz_convert(TZ)
    return expanded


def build_hourly_series(alerts: pd.DataFrame) -> pd.DataFrame:
    expanded = expand_to_hours(alerts)

    # per-hour aggregation: alert=1 if any alert touched that hour
    agg = (
        expanded.groupby("hour")
        .agg(alert=("naive", "count"), naive_flag=("naive", "max"))
        .reset_index()
    )
    agg["alert"] = 1  # every row in agg means at least one alert was active

    # full hourly index: start at first alert, end at last complete hour of July
    # (cutoff midnight UTC = 03:00 Kyiv; floor to day - 1h = 23:00 July 31 Kyiv)
    cutoff_local = pd.Timestamp(CUTOFF_UTC).tz_convert(TZ)
    series_end = cutoff_local.floor("D") - pd.Timedelta("1h")
    full_index = pd.date_range(
        start=alerts["started_at"].min().floor("h"),
        end=series_end,
        freq="h",
        tz=TZ,
    )
    hourly = pd.DataFrame({"hour": full_index})
    hourly = hourly.merge(agg[["hour", "alert", "naive_flag"]], on="hour", how="left")
    hourly["alert"] = hourly["alert"].fillna(0).astype(int)
    hourly["naive_flag"] = hourly["naive_flag"].fillna(0).astype(int)

    return hourly


if __name__ == "__main__":
    alerts = load_kharkiv_alerts()
    print(f"Kharkiv oblast alerts loaded: {len(alerts)}")
    print(f"Naive (30-min imputed): {alerts['naive'].sum()}")

    hourly = build_hourly_series(alerts)
    print(f"\nHourly series: {len(hourly)} rows")
    print(f"Alert rate: {hourly['alert'].mean():.3f} ({hourly['alert'].sum()} alert-hours)")
    print(f"Date range: {hourly['hour'].min()} → {hourly['hour'].max()}")

    hourly.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
