"""Shared evaluation utilities used by baseline.py and train_lgbm.py."""
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)


def print_metrics(name: str, y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true)

    pr_auc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    base_rate = y_true.mean()

    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(f"  PR-AUC   : {pr_auc:.4f}  (base rate: {base_rate:.4f})")
    print(f"  F1       : {f1:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"{'─' * 40}")

    return dict(name=name, pr_auc=pr_auc, f1=f1, precision=prec, recall=rec)
