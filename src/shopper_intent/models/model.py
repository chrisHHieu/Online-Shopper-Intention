"""Fitted model artifact: estimator, calibrator, threshold and conformal policy.

The object is what ``model.joblib`` contains. It knows how to turn a feature
frame into a calibrated probability, a business decision, a conformal band and
per-row contributions.

The stored feature contract is the fixed one-hot matrix (32 columns). CatBoost
and XGBoost are fitted on the native view built from that matrix, so the
processed table, the API and the drift check all keep one column contract.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from shopper_intent.features.build import to_native


class IntentModel:
    """Estimator plus calibration plus threshold, serialisable with joblib."""

    # Class level defaults keep artifacts written before these fields existed
    # loadable: a missing instance attribute resolves here.
    categorical_vocabulary: dict[str, list[Any]] | None = None
    native_categorical: bool = False
    conformal: Any = None
    explain_engine: str | None = None

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
        categorical_vocabulary: dict[str, list[Any]] | None = None,
        native_categorical: bool = False,
        conformal: Any = None,
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
        self.categorical_vocabulary = (
            dict(categorical_vocabulary) if categorical_vocabulary else None
        )
        self.native_categorical = bool(native_categorical)
        self.conformal = conformal

    # ---------------------------------------------------------------- scoring
    def _select(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = [column for column in self.feature_names if column not in frame.columns]
        if missing:
            raise ValueError(f"missing feature columns: {missing}")
        return frame[self.feature_names]

    def _matrix(self, frame: pd.DataFrame | np.ndarray) -> pd.DataFrame | np.ndarray:
        if not isinstance(frame, pd.DataFrame):
            return frame
        selected = self._select(frame)
        if self.native_categorical and self.categorical_vocabulary:
            return to_native(
                selected,
                self.categorical_vocabulary,
                as_category=self.model_type == "xgboost",
            )
        return selected

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
            "band": self.band(probability),
        }

    def band(self, probability: np.ndarray) -> np.ndarray:
        """Conformal band per row, or ``unsupported`` when no policy is stored."""
        if self.conformal is None:
            return np.full(len(np.asarray(probability)), "unsupported", dtype=object)
        return self.conformal.bands(probability)

    def coverage(self, target: np.ndarray, probability: np.ndarray) -> float | None:
        if self.conformal is None:
            return None
        return float(self.conformal.coverage(target, probability))

    # ---------------------------------------------------------- explainability
    def explain(self, frame: pd.DataFrame, top_k: int = 3) -> list[list[dict[str, Any]]]:
        """Top-k additive contributions per row, largest absolute value first."""
        from shopper_intent.models.explain import contribution_matrix, top_reasons

        matrix = self._matrix(frame)
        _, engine = contribution_matrix(self, matrix)
        self.explain_engine = engine
        reasons = top_reasons(
            self,
            matrix,
            top_k=top_k,
            vocabulary=self.categorical_vocabulary,
        )
        return reasons

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
            "calibration": self.calibrator.info() if self.calibrator is not None else None,
            "conformal": self.conformal.info() if self.conformal is not None else None,
            "explain_engine": self.explain_engine,
            "native_categorical": self.native_categorical,
        }
