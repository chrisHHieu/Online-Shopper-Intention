"""Population stability index against the training reference.

The reference distribution is the training split of the processed table. The
current distribution defaults to the held-out split, which is a sanity check
rather than a real monitoring signal; pass ``--current`` with a raw CSV of new
sessions to score a real batch.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shopper_intent.config import Settings
from shopper_intent.features.build import prepare_for_model, read_feature_schema, read_processed
from shopper_intent.models.registry import utc_now_iso

STABLE = 0.10
MODERATE = 0.25
CONSTANT_SHIFT_PSI = 10.0


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI with quantile bins from the reference distribution and epsilon smoothing."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected = expected[np.isfinite(expected)]
    actual = actual[np.isfinite(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    unique = np.unique(expected)
    if len(unique) == 1:
        share = float(np.mean(actual == unique[0]))
        return 0.0 if share == 1.0 else CONSTANT_SHIFT_PSI
    if len(unique) == 2 and set(unique.tolist()) <= {0.0, 1.0}:
        edges = np.array([-np.inf, 0.5, np.inf])
    else:
        quantiles = np.linspace(0, 1, bins + 1)
        edges = np.unique(np.quantile(expected, quantiles))
        if len(edges) < 3:
            midpoint = float(np.mean([unique[0], unique[-1]]))
            edges = np.array([-np.inf, midpoint, np.inf])
        else:
            edges[0] = -np.inf
            edges[-1] = np.inf
    expected_counts = np.histogram(expected, bins=edges)[0].astype(float)
    actual_counts = np.histogram(actual, bins=edges)[0].astype(float)
    epsilon = 1e-6
    expected_share = np.maximum(expected_counts / expected_counts.sum(), epsilon)
    actual_share = np.maximum(actual_counts / actual_counts.sum(), epsilon)
    return float(np.sum((actual_share - expected_share) * np.log(actual_share / expected_share)))


def status_for(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    if value < STABLE:
        return "stable"
    if value < MODERATE:
        return "moderate"
    return "significant"


def drift_rows(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    features: list[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        if feature not in reference.columns or feature not in current.columns:
            continue
        value = psi(reference[feature].to_numpy(), current[feature].to_numpy())
        ref_mean = float(reference[feature].mean())
        cur_mean = float(current[feature].mean())
        ref_std = float(reference[feature].std(ddof=0))
        shift = (cur_mean - ref_mean) / ref_std if ref_std > 0 else 0.0
        rows.append(
            {
                "feature": feature,
                "psi": value,
                "status": status_for(value),
                "reference_mean": ref_mean,
                "current_mean": cur_mean,
                "mean_shift_in_sd": float(shift),
            }
        )
    rows.sort(key=lambda row: (-row["psi"] if np.isfinite(row["psi"]) else 0.0))
    return rows


def current_frame(config: Settings, current_path: Path | None) -> tuple[pd.DataFrame, str]:
    if current_path is None:
        frame = read_processed(config)
        test = frame[frame["split"] == "test"].reset_index(drop=True)
        features = read_feature_schema(config)["features"]
        return test[features], "held_out_test_split"
    raw = pd.read_csv(current_path)
    return prepare_for_model(raw, config), str(current_path)


def drift(config: Settings, current_path: Path | None = None) -> dict[str, Any]:
    """Compare the current distribution with the training reference and write a report."""
    frame = read_processed(config)
    features = read_feature_schema(config)["features"]
    reference = frame[frame["split"] == "train"].reset_index(drop=True)
    current, source = current_frame(config, current_path)
    rows = drift_rows(reference[features], current, features)
    significant = [row for row in rows if row["status"] == "significant"]
    report = {
        "generated_at": utc_now_iso(),
        "reference": "training split",
        "current": source,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "thresholds": {"stable_below": STABLE, "significant_from": MODERATE},
        "summary": {
            "features_checked": len(rows),
            "significant": len(significant),
            "moderate": sum(1 for row in rows if row["status"] == "moderate"),
            "max_psi": float(rows[0]["psi"]) if rows else float("nan"),
            "drift_detected": bool(significant),
        },
        "features": rows,
    }
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "drift.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
