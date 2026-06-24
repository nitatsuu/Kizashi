"""
Naive baselines evaluated on the test set (Jan 2025 – Jul 2025).

Baselines:
  - Persistence : predict alert[T+1] = alert[T]  (uses alert_lag_1 as proxy)
  - Majority    : always predict 1 (alert)
Run from project root: python src/baseline.py
"""
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from evaluate import print_metrics

DATA_PATH = "data/kharkiv_features.parquet"
TEST_START = "2025-01-01"


def load_test() -> pd.DataFrame:
    df = pd.read_parquet(DATA_PATH)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    return df[df["hour"] >= TEST_START].reset_index(drop=True)


if __name__ == "__main__":
    test = load_test()
    y_true = test["target"].values
    print(f"Test set: {len(test)} hours | base rate: {y_true.mean():.3f}")

    # Persistence: predict next hour = current hour; alert column is alert[T]
    # (alert_lag_1 = alert[T-1] — using that would be an off-by-one error)
    print_metrics("Persistence (alert[T] → alert[T+1])", y_true, test["alert"].values)

    # Majority: always predict 1
    print_metrics("Majority class (always alert=1)", y_true, [1.0] * len(y_true))
