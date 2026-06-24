"""
LightGBM binary classifier for next-hour alert prediction.
Trains on two feature sets (with / without days_since_start) and reports metrics.

Temporal split:
  train : Mar 2022 – Sep 2024
  val   : Oct 2024 – Dec 2024  (early stopping only, not reported)
  test  : Jan 2025 – Jul 2025

Run from project root: python src/train_lgbm.py
"""
import os
import sys
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(__file__))
from evaluate import print_metrics

FEATURES_WITH_DRIFT    = "data/kharkiv_features.parquet"
FEATURES_NO_DRIFT      = "data/kharkiv_features_no_drift.parquet"
MODEL_DIR              = "models"
TRAIN_END              = "2024-09-30"
VAL_END                = "2024-12-31"
TEST_START             = "2025-01-01"

FEATURE_COLS_DRIFT = [
    "alert_lag_1", "alert_lag_2", "alert_lag_3",
    "alert_lag_24", "alert_lag_48",
    "alert_roll_3h", "alert_roll_6h", "alert_roll_24h",
    "hour_of_day", "day_of_week", "month", "is_weekend",
    "days_since_start",
]
FEATURE_COLS_NO_DRIFT = [c for c in FEATURE_COLS_DRIFT if c != "days_since_start"]

LGBM_PARAMS = dict(
    objective="binary",
    metric="average_precision",
    learning_rate=0.05,
    num_leaves=31,
    max_depth=6,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=1000,
    random_state=42,
    verbose=-1,
)


def split(df: pd.DataFrame):
    hour = pd.to_datetime(df["hour"], utc=True)
    train = df[hour <= TRAIN_END]
    val   = df[(hour > TRAIN_END) & (hour <= VAL_END)]
    test  = df[hour >= TEST_START]
    return train, val, test


def train_and_evaluate(data_path: str, feature_cols: list, label: str):
    df = pd.read_parquet(data_path)
    df["hour"] = pd.to_datetime(df["hour"], utc=True)
    train, val, test = split(df)

    X_train, y_train = train[feature_cols], train["target"]
    X_val,   y_val   = val[feature_cols],   val["target"]
    X_test,  y_test  = test[feature_cols],  test["target"]

    print(f"\n{'=' * 50}")
    print(f"  {label}")
    print(f"  train={len(train)}  val={len(val)}  test={len(test)}")
    print(f"{'=' * 50}")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"  Best iteration: {model.best_iteration_}")

    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = print_metrics(label, y_test, y_prob)

    # save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_name = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
    model_path = f"{MODEL_DIR}/{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Model saved → {model_path}")

    # feature importance: gain (information per split) and split count (usage frequency)
    imp_gain  = pd.Series(model.booster_.feature_importance(importance_type="gain"),  index=feature_cols)
    imp_split = pd.Series(model.booster_.feature_importance(importance_type="split"), index=feature_cols)
    order = imp_gain.sort_values().index  # sort both by gain for easy comparison

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    imp_gain[order].plot.barh(ax=axes[0], color="steelblue")
    axes[0].set_title("Importance by Gain\n(information contributed per feature)")
    axes[0].set_xlabel("Total gain")

    imp_split[order].plot.barh(ax=axes[1], color="darkorange")
    axes[1].set_title("Importance by Split count\n(how often the feature was used)")
    axes[1].set_xlabel("Number of splits")

    fig.suptitle(f"Feature importance — {label}", y=1.01)
    fig.tight_layout()
    fig_path = f"figures/lgbm_importance_{model_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Importance plot → {fig_path}")

    return metrics


if __name__ == "__main__":
    results = []
    results.append(train_and_evaluate(
        FEATURES_WITH_DRIFT, FEATURE_COLS_DRIFT, "LightGBM (with days_since_start)"
    ))
    results.append(train_and_evaluate(
        FEATURES_NO_DRIFT, FEATURE_COLS_NO_DRIFT, "LightGBM (no days_since_start)"
    ))

    print("\n\n── Summary ──────────────────────────────────")
    print(f"{'Model':<40} {'PR-AUC':>8} {'F1':>8} {'Recall':>8}")
    print("─" * 66)
    for r in results:
        print(f"{r['name']:<40} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['recall']:>8.4f}")
