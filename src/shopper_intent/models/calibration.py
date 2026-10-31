"""Post-hoc probability calibration with an out-of-fold choice of method.

Two methods are supported: isotonic regression, which is flexible and can
overfit when a bin holds few rows, and Platt scaling, which fits a one
dimensional logistic map and is therefore much more constrained. ``auto`` fits
both on the out-of-fold training scores and keeps whichever has the lower
out-of-fold Brier score. The held-out split never takes part in the choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

METHODS = ("isotonic", "platt")


@dataclass
class Calibrator:
    """Serialisable probability map fitted on out-of-fold scores.

    ``predict`` keeps the same signature the earlier isotonic object had, so
    artifacts written before this module existed still load and score.
    """

    method: str
    transforms: dict[str, Any] = field(default_factory=dict)
    clip_low: float = 1e-4
    clip_high: float = 1 - 1e-4
    requested: str = "isotonic"
    selection: dict[str, float] = field(default_factory=dict)

    def predict(self, raw: np.ndarray) -> np.ndarray:
        raw = np.asarray(raw, dtype=float)
        transform = self.transforms[self.method]
        if self.method == "isotonic":
            values = transform.predict(raw)
        else:
            values = transform.predict_proba(raw.reshape(-1, 1))[:, 1]
        return np.clip(np.asarray(values, dtype=float), self.clip_low, self.clip_high)

    def info(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "requested": self.requested,
            "candidates": sorted(self.transforms),
            "selection": dict(self.selection),
            "clip_low": self.clip_low,
            "clip_high": self.clip_high,
        }


def _fit_isotonic(config: Any, raw: np.ndarray, target: np.ndarray) -> IsotonicRegression:
    model = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=config.calibration.clip_low,
        y_max=config.calibration.clip_high,
    )
    model.fit(raw, target)
    return model


def _fit_platt(config: Any, raw: np.ndarray, target: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    model.fit(np.asarray(raw, dtype=float).reshape(-1, 1), target)
    return model


def _score(method: str, transform: Any, raw: np.ndarray, target: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        values = transform.predict(raw)
    else:
        values = transform.predict_proba(raw.reshape(-1, 1))[:, 1]
    return np.clip(np.asarray(values, dtype=float), 1e-9, 1 - 1e-9)


def fit_calibrator(
    config: Any,
    raw_probabilities: np.ndarray,
    target: np.ndarray,
) -> Calibrator:
    """Fit the configured calibrator (or both, for ``auto``) on OOF scores."""
    raw = np.asarray(raw_probabilities, dtype=float)
    target = np.asarray(target).astype(int)
    requested = config.calibration.method
    wanted = METHODS if requested == "auto" else (requested,)

    transforms: dict[str, Any] = {}
    brier: dict[str, float] = {}
    for method in wanted:
        fitted = _fit_isotonic(config, raw, target) if method == "isotonic" else _fit_platt(
            config, raw, target
        )
        transforms[method] = fitted
        brier[method] = float(brier_score_loss(target, _score(method, fitted, raw, target)))

    chosen = min(brier, key=lambda name: (brier[name], name != "isotonic"))
    return Calibrator(
        method=chosen,
        transforms=transforms,
        clip_low=config.calibration.clip_low,
        clip_high=config.calibration.clip_high,
        requested=requested,
        selection={f"brier_{method}": value for method, value in brier.items()},
    )


def apply_calibrator(calibrator: Any, raw: np.ndarray, clip_low: float, clip_high: float):
    """Apply any calibrator that implements ``predict`` and clip the result."""
    if calibrator is None:
        return np.asarray(raw, dtype=float)
    values = np.asarray(calibrator.predict(raw), dtype=float)
    return np.clip(values, clip_low, clip_high)


def reliability_curve(
    target: np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float]]:
    """Quantile binned predicted against observed rate, for the model card."""
    y = np.asarray(target).astype(int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, bins + 1)))
    rows: list[dict[str, float]] = []
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        mask = (p >= lower) & (p < upper)
        if mask.sum() == 0:
            continue
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return rows
