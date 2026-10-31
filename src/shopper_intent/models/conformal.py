"""Split conformal prediction sets for the two class conversion problem.

The calibration scores are ``1 - p(y)`` evaluated on out-of-fold training
probabilities, so every score comes from a model that did not see that row.
``q_hat`` is the finite sample split conformal quantile at level ``1 - alpha``.

A prediction set holds the labels whose nonconformity score is at or below
``q_hat``. With two classes that gives three business bands:

``skip``      the set is ``{0}`` only: a conversion is implausible at this level.
``standard``  the set holds both labels or none: the session is ambiguous, so it
              is neither confidently discarded nor confidently prioritised.
``priority`` the set is ``{1}`` only: a conversion is plausible and non conversion
              is not.

Coverage is the fraction of the held-out rows whose true label is inside its
set, measured once on data that took no part in fitting the model or ``q_hat``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

BANDS = ("skip", "standard", "priority")


@dataclass
class ConformalPolicy:
    alpha: float
    q_hat: float
    calibration_rows: int
    target_coverage: float

    def _contains(self, probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(probability, dtype=float)
        negative = p <= self.q_hat
        positive = (1.0 - p) <= self.q_hat
        return negative, positive

    def bands(self, probability: np.ndarray) -> np.ndarray:
        negative, positive = self._contains(probability)
        labels = np.full(len(negative), "standard", dtype=object)
        labels[negative & ~positive] = "skip"
        labels[positive & ~negative] = "priority"
        return labels

    def prediction_sets(self, probability: np.ndarray) -> list[list[int]]:
        negative, positive = self._contains(probability)
        sets: list[list[int]] = []
        for keep_negative, keep_positive in zip(negative, positive, strict=True):
            labels = []
            if keep_negative:
                labels.append(0)
            if keep_positive:
                labels.append(1)
            sets.append(labels)
        return sets

    def coverage(self, target: np.ndarray, probability: np.ndarray) -> float:
        y = np.asarray(target).astype(int)
        negative, positive = self._contains(probability)
        inside = np.where(y == 1, positive, negative)
        return float(np.mean(inside)) if len(y) else 0.0

    def info(self) -> dict[str, Any]:
        return {
            "method": "split conformal",
            "alpha": float(self.alpha),
            "target_coverage": float(self.target_coverage),
            "q_hat": float(self.q_hat),
            "calibration_rows": int(self.calibration_rows),
            "bands": list(BANDS),
        }


def fit_conformal(
    configuration: Any,
    target: np.ndarray,
    probabilities: np.ndarray,
) -> ConformalPolicy | None:
    """Fit ``q_hat`` on out-of-fold calibrated probabilities, or return None."""
    if not configuration.conformal.enabled:
        return None
    y = np.asarray(target).astype(int)
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-9, 1 - 1e-9)
    if len(y) == 0:
        return None
    alpha = float(configuration.conformal.alpha)
    scores = 1.0 - np.where(y == 1, p, 1.0 - p)
    rows = len(scores)
    level = min(1.0, float(np.ceil((rows + 1) * (1.0 - alpha))) / rows)
    q_hat = float(np.quantile(scores, level, method="higher"))
    return ConformalPolicy(
        alpha=alpha,
        q_hat=min(1.0, max(0.0, q_hat)),
        calibration_rows=rows,
        target_coverage=1.0 - alpha,
    )


def band_report(
    policy: ConformalPolicy,
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    """Coverage, set sizes and observed conversion rate per band on one split."""
    y = np.asarray(target).astype(int)
    bands = policy.bands(probabilities)
    sets = policy.prediction_sets(probabilities)
    rows: dict[str, dict[str, float]] = {}
    for band in BANDS:
        mask = bands == band
        rows[band] = {
            "sessions": int(mask.sum()),
            "share": float(mask.mean()) if len(mask) else 0.0,
            "observed_conversion_rate": float(y[mask].mean()) if mask.sum() else 0.0,
        }
    return {
        "alpha": float(policy.alpha),
        "target_coverage": float(policy.target_coverage),
        "q_hat": float(policy.q_hat),
        "coverage": policy.coverage(y, probabilities),
        "mean_set_size": float(np.mean([len(item) for item in sets])) if sets else 0.0,
        "empty_fraction": float(np.mean([len(item) == 0 for item in sets])) if sets else 0.0,
        "bands": rows,
    }
