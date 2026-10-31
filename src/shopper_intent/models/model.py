"""Fitted model artifact: estimator, isotonic calibrator and decision policy.

The object is what ``model.joblib`` contains. It knows how to turn a feature
frame into a calibrated probability, a business decision, and per-row tree SHAP
contributions for the LightGBM model (linear contributions for logistic
regression, which are exact contributions to the logit).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


class IntentModel:
    """Estimator plus calibration plus threshold, serialisable with joblib."""

    def __init__(
        self,
        estimator: Pipeline,
        calibrator: Any,
        feature_names: list[str],
        threshold: float,
        value_per_conversion: float,
        cost_per_outreach: float,
        model_type: str,
        version: str,
        trained_at: str,
        config_hash: str,
        data_sha256: str,
        clip_low: float = 1e-4,
        clip_high: float = 1 - 1e-4,
    ) -> None:
        self.estimator = estimator
        self.calibrator = calibrator
        self.feature_names = list(feature_names)
        self.threshold = float(threshold)
        self.value_per_conversion = float(value_per_conversion)
        self.cost_per_outreach = float(cost_per_outreach)
        self.model_type = model_type
        self.version = version
        self.trained_at = trained_at
        self.config_hash = config_hash
        self.data_sha256 = data_sha256
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)

    # ---------------------------------------------------------------- scoring
    def _matrix(self, frame: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if isinstance(frame, pd.DataFrame):
            missing = [column for column in self.feature_names if column not in frame.columns]
            if missing:
                raise ValueError(f"missing feature columns: {missing}")
            return frame[self.feature_names]
        return frame

    def _preprocess(self, matrix: pd.DataFrame | np.ndarray) -> np.ndarray:
        steps = self.estimator.steps
        if len(steps) > 1:
            transformed = Pipeline(steps[:-1]).transform(matrix)
        else:
            transformed = np.asarray(matrix, dtype=float)
        return np.asarray(transformed, dtype=float)

    def predict_proba_raw(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        matrix = self._matrix(frame)
        proba = self.estimator.predict_proba(matrix)[:, 1]
        return np.clip(np.asarray(proba, dtype=float), 1e-9, 1 - 1e-9)

    def predict_proba(self, frame: pd.DataFrame | np.ndarray) -> np.ndarray:
        raw = self.predict_proba_raw(frame)
        if self.calibrator is None:
            return raw
        calibrated = np.asarray(self.calibrator.predict(raw), dtype=float)
        return np.clip(calibrated, self.clip_low, self.clip_high)

    def expected_value(self, probability: np.ndarray) -> np.ndarray:
        return np.asarray(probability) * self.value_per_conversion - self.cost_per_outreach

    def predict(self, frame: pd.DataFrame | np.ndarray) -> dict[str, Any]:
        probability = self.predict_proba(frame)
        net_if_contacted = self.expected_value(probability)
        decision = np.where(probability >= self.threshold, "contact", "skip")
        net_of_decision = np.where(decision == "contact", np.maximum(net_if_contacted, 0.0), 0.0)
        return {
            "probability": probability,
            "decision": decision,
            "expected_value_if_contacted": net_if_contacted,
            "expected_value_of_decision": net_of_decision,
        }

    # ---------------------------------------------------------- explainability
    def explain(self, frame: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
        """Top-k additive contributions per row, largest absolute value first."""
        matrix = self._matrix(frame)
        transformed = self._preprocess(matrix)
        final = self.estimator.steps[-1][1]
        if hasattr(final, "booster_"):
            contribs = final.booster_.predict(transformed, pred_contrib=True)
            contribs = np.asarray(contribs, dtype=float)[:, : len(self.feature_names)]
        elif hasattr(final, "coef_"):
            contribs = transformed * final.coef_[0]
        else:
            raise TypeError(f"model type {self.model_type!r} does not support contributions")

        explanations: list[list[dict[str, Any]]] = []
        for row in contribs:
            order = np.argsort(-np.abs(row))[:top_k]
            explanations.append(
                [
                    {
                        "feature": self.feature_names[index],
                        "contribution": round(float(row[index]), 5),
                        "direction": "increases" if row[index] >= 0 else "decreases",
                    }
                    for index in order
                ]
            )
        return explanations

    # ------------------------------------------------------------------ info
    def info(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "version": self.version,
            "trained_at": self.trained_at,
            "threshold": self.threshold,
            "value_per_conversion": self.value_per_conversion,
            "cost_per_outreach": self.cost_per_outreach,
            "n_features": len(self.feature_names),
            "config_hash": self.config_hash,
            "data_sha256": self.data_sha256,
            "supports_contributions": True,
        }
