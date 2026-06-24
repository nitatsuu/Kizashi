"""
EDA for Kharkiv hourly alert series.
Produces figures/ *.png used in the write-up.
Run from project root: python src/eda.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

DATA_PATH = "data/kharkiv_hourly.parquet"
FIG_DIR = "figures"


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"], utc=True).dt.tz_convert("Europe/Kyiv")
    return df


# ── Figure 1: monthly alert rate (drift) ────────────────────────────────────

def fig_monthly_rate(df: pd.DataFrame) -> None:
    monthly = df.set_index("hour")["alert"].resample("ME").mean()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly.index, monthly.values, width=20, color="steelblue", alpha=0.8)
    ax.axhline(df["alert"].mean(), color="firebrick", lw=1.2, ls="--", label=f"overall mean {df['alert'].mean():.2f}")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45, ha="right")
    ax.set_ylabel("Fraction of hours with alert")
    ax.set_title("Kharkiv – monthly alert rate (concept drift check)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/01_monthly_alert_rate.png", dpi=150)
    plt.close(fig)
    print("Saved 01_monthly_alert_rate.png")


# ── Figure 2: hour-of-day × day-of-week heatmap ─────────────────────────────

def fig_seasonality(df: pd.DataFrame) -> None:
    df = df.copy()
    df["hour_of_day"] = df["hour"].dt.hour
    df["dow"] = df["hour"].dt.day_of_week  # 0=Mon … 6=Sun

    pivot = df.groupby(["dow", "hour_of_day"])["alert"].mean().unstack("hour_of_day")
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_yticks(range(7))
    ax.set_yticklabels(dow_labels)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels(range(0, 24, 2))
    ax.set_xlabel("Hour of day (Kyiv time)")
    ax.set_title("Alert rate by hour × weekday")
    fig.colorbar(im, ax=ax, label="alert rate")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/02_seasonality_heatmap.png", dpi=150)
    plt.close(fig)
    print("Saved 02_seasonality_heatmap.png")


# ── Figure 3: STL decomposition (daily alert rate) ───────────────────────────

def fig_stl(df: pd.DataFrame) -> None:
    daily = df.set_index("hour")["alert"].resample("D").mean().dropna()

    # period=7 for weekly seasonality
    stl = STL(daily, period=7, robust=True)
    res = stl.fit()

    fig = res.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle("STL decomposition of daily alert rate (period=7 days)", y=1.01)
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/03_stl_decomposition.png", dpi=150)
    plt.close(fig)
    print("Saved 03_stl_decomposition.png")


# ── Figure 4: ACF / PACF of hourly series ───────────────────────────────────

def fig_acf(df: pd.DataFrame) -> None:
    series = df["alert"].values
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))

    plot_acf(series, lags=48, ax=axes[0], title="ACF – hourly alert (lags 0–48h)")
    plot_pacf(series, lags=48, ax=axes[1], title="PACF – hourly alert (lags 0–48h)", method="ywm")

    for ax in axes:
        ax.axvline(24, color="steelblue", lw=0.8, ls="--", alpha=0.6, label="lag 24h")
        ax.axvline(48, color="steelblue", lw=0.8, ls="--", alpha=0.6, label="lag 48h")

    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/04_acf_pacf.png", dpi=150)
    plt.close(fig)
    print("Saved 04_acf_pacf.png")


if __name__ == "__main__":
    df = load()
    print(f"Loaded {len(df)} rows | base rate: {df['alert'].mean():.3f}")
    fig_monthly_rate(df)
    fig_seasonality(df)
    fig_stl(df)
    fig_acf(df)
    print("\nAll figures saved to figures/")
