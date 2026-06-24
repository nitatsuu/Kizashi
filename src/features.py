"""
Build feature matrix from hourly alert series.
All features use only information available at time T to predict alert at T+1.
Output: data/kharkiv_features.parquet
"""
import pandas as pd

DATA_PATH = "data/kharkiv_hourly.parquet"
OUT_PATH = "data/kharkiv_features.parquet"
TZ = "Europe/Kyiv"


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert(TZ)
    return df.sort_values("hour").reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    alert = df["alert"]
    df["alert_lag_1"] = alert.shift(1)
    df["alert_lag_2"] = alert.shift(2)
    df["alert_lag_3"] = alert.shift(3)
    df["alert_lag_24"] = alert.shift(24)
    df["alert_lag_48"] = alert.shift(48)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    alert = df["alert"].astype(float)
    # window includes T (the current hour) — valid when predicting T+1
    df["alert_roll_3h"] = alert.rolling(3).mean()
    df["alert_roll_6h"] = alert.rolling(6).mean()
    df["alert_roll_24h"] = alert.rolling(24).mean()
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    # calendar features for the TARGET hour (T+1), not the current hour
    target_hour = df["hour"] + pd.Timedelta("1h")
    df["hour_of_day"] = target_hour.dt.hour
    df["day_of_week"] = target_hour.dt.dayofweek   # 0=Mon, 6=Sun
    df["month"] = target_hour.dt.month
    df["is_weekend"] = (target_hour.dt.dayofweek >= 5).astype(int)
    return df


def add_drift_feature(df: pd.DataFrame) -> pd.DataFrame:
    origin = df["hour"].min()
    df["days_since_start"] = (df["hour"] - origin).dt.total_seconds() / 86400
    df["days_since_start"] = df["days_since_start"].round(4)
    return df


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    df["target"] = df["alert"].shift(-1).astype("Int64")
    return df


if __name__ == "__main__":
    df = load()

    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_drift_feature(df)
    df = add_target(df)

    # drop rows where lags or target are NaN (first 48 rows + last 1 row)
    before = len(df)
    df = df.dropna().reset_index(drop=True)
    print(f"Rows after dropping NaN lag/target: {len(df)} (dropped {before - len(df)})")

    feature_cols = [
        "alert_lag_1", "alert_lag_2", "alert_lag_3",
        "alert_lag_24", "alert_lag_48",
        "alert_roll_3h", "alert_roll_6h", "alert_roll_24h",
        "hour_of_day", "day_of_week", "month", "is_weekend",
        "days_since_start",
    ]

    print(f"\nFeature set ({len(feature_cols)} features):")
    for f in feature_cols:
        print(f"  {f}: min={df[f].min():.3f}  max={df[f].max():.3f}")

    print(f"\nTarget distribution:\n{df['target'].value_counts().to_dict()}")
    print(f"Base rate: {df['target'].mean():.3f}")

    df[["hour", "alert", *feature_cols, "target", "naive_flag"]].to_parquet(OUT_PATH, index=False)
    print(f"\nSaved → {OUT_PATH}")
