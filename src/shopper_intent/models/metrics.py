"""Metrics, decision threshold selection and the decile lift table.

The threshold is chosen on out-of-fold training probabilities by maximising
expected net value: value per conversion times true positives minus cost per
outreach times every contacted session. It is never chosen on the test split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def ks_statistic(y_true: np.ndarray, proba: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(y_true, proba)
    return float(np.max(tpr - fpr))


def value_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    value_per_conversion: float,
    cost_per_outreach: float,
) -> dict[str, Any]:
    """Confusion counts and net value for one decision threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    decision = p >= threshold
    tp = int(np.sum((y == 1) & decision))
    fp = int(np.sum((y == 0) & decision))
    fn = int(np.sum((y == 1) & ~decision))
    tn = int(np.sum((y == 0) & ~decision))
    contacted = tp + fp
    positives = tp + fn
    net = tp * value_per_conversion - contacted * cost_per_outreach
    return {
        "threshold": float(threshold),
        "contacted": contacted,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "precision": float(tp / contacted) if contacted else 0.0,
        "recall": float(tp / positives) if positives else 0.0,
        "conversions_captured": tp,
        "value_created": float(tp * value_per_conversion),
        "outreach_cost": float(contacted * cost_per_outreach),
        "net_value": float(net),
        "net_value_per_1000_sessions": float(net / len(y) * 1000.0) if len(y) else 0.0,
        "contact_rate": float(contacted / len(y)) if len(y) else 0.0,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    value_per_conversion: float,
    cost_per_outreach: float,
) -> dict[str, Any]:
    """Full metric block for one model at one threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    decision = p >= threshold
    matrix = confusion_matrix(y, decision, labels=[0, 1])
    tn, fp, fn, tp = (int(value) for value in matrix.ravel())
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    values = value_summary(y, p, threshold, value_per_conversion, cost_per_outreach)
    base_metrics = {
        "precision": float(precision_score(y, decision, zero_division=0)),
        "recall": float(recall_score(y, decision, zero_division=0)),
        "f1": float(f1_score(y, decision, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float((tp + tn) / len(y)) if len(y) else 0.0,
        "ks_stat": ks_statistic(y, p),
        "ks_at_threshold": float(tpr - fpr),
        "true_positive_rate_at_threshold": float(tpr),
        "false_positive_rate_at_threshold": float(fpr),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }
    return {**values, **base_metrics}


def threshold_candidates(proba: np.ndarray, limit: int) -> np.ndarray:
    """Candidate thresholds from the observed probability mass, always including 0.5."""
    values = np.unique(np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6))
    if len(values) > limit:
        positions = np.linspace(0, len(values) - 1, limit).round().astype(int)
        values = values[positions]
    return np.unique(np.append(values, 0.5))


def choose_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    value_per_conversion: float,
    cost_per_outreach: float,
    limit: int = 500,
) -> dict[str, Any]:
    """Pick the threshold with the highest net value; ties go to the higher threshold."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    candidates = threshold_candidates(p, limit)
    rows = [
        value_summary(y, p, float(threshold), value_per_conversion, cost_per_outreach)
        for threshold in candidates
    ]
    best = max(rows, key=lambda row: (round(row["net_value"], 6), row["threshold"]))
    return {
        "threshold": float(best["threshold"]),
        "net_value": float(best["net_value"]),
        "net_value_per_1000_sessions": float(best["net_value_per_1000_sessions"]),
        "contact_rate": float(best["contact_rate"]),
        "evaluated_thresholds": int(len(rows)),
    }


def threshold_sweep(
    y_true: np.ndarray,
    proba: np.ndarray,
    value_per_conversion: float,
    cost_per_outreach: float,
    points: int = 200,
) -> list[dict[str, float]]:
    thresholds = np.linspace(1e-4, 0.999, points)
    return [
        {
            "threshold": float(threshold),
            "net_value_per_1000_sessions": float(
                value_summary(
                    y_true,
                    proba,
                    float(threshold),
                    value_per_conversion,
                    cost_per_outreach,
                )["net_value_per_1000_sessions"]
            ),
        }
        for threshold in thresholds
    ]


def decile_lift(y_true: np.ndarray, proba: np.ndarray, bins: int = 10) -> list[dict[str, Any]]:
    """Sort by predicted probability and report response rate and lift per bin."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    order = np.argsort(-p, kind="mergesort")
    prevalence = float(y.mean()) if len(y) else 0.0
    total_positives = int(y.sum())
    rows: list[dict[str, Any]] = []
    captured = 0
    for index, chunk in enumerate(np.array_split(order, bins), start=1):
        positives = int(y[chunk].sum())
        captured += positives
        response = float(positives / len(chunk)) if len(chunk) else 0.0
        rows.append(
            {
                "decile": index,
                "sessions": int(len(chunk)),
                "positives": positives,
                "response_rate": response,
                "lift": float(response / prevalence) if prevalence else 0.0,
                "cumulative_positives": captured,
                "cumulative_capture": float(captured / total_positives)
                if total_positives
                else 0.0,
                "min_score": float(p[chunk].min()) if len(chunk) else 0.0,
                "max_score": float(p[chunk].max()) if len(chunk) else 0.0,
            }
        )
    return rows


def policy_summary(
    y_true: np.ndarray,
    proba: np.ndarray,
    threshold: float,
    default_threshold: float,
    value_per_conversion: float,
    cost_per_outreach: float,
) -> dict[str, Any]:
    """Compare the chosen policy with the 0.5 default and the always-contact policy."""
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    chosen = value_summary(y, p, threshold, value_per_conversion, cost_per_outreach)
    default = value_summary(y, p, default_threshold, value_per_conversion, cost_per_outreach)
    always = value_summary(y, p, 0.0, value_per_conversion, cost_per_outreach)
    return {
        "chosen": chosen,
        "default_threshold": default,
        "always_contact": always,
        "gap_vs_default_per_1000_sessions": float(
            chosen["net_value_per_1000_sessions"] - default["net_value_per_1000_sessions"]
        ),
        "gap_vs_always_contact_per_1000_sessions": float(
            chosen["net_value_per_1000_sessions"] - always["net_value_per_1000_sessions"]
        ),
    }
