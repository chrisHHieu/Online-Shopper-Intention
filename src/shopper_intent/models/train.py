"""Training entry point.

Protocol: one stratified 80/20 split. Inside the training split, stratified
k-fold cross-validation produces out-of-fold probabilities. Those are used for
every headline number, for fitting the isotonic calibrator, and for choosing the
business threshold. The held-out split is scored once and only reports.

The calibrator is fitted on out-of-fold predictions from models trained on 4/5
of the training split and then applied to a model refitted on all of it. This is
the usual compromise; the held-out split is where calibration quality is judged.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shopper_intent.config import Settings
from shopper_intent.data.fetch import sha256_file
from shopper_intent.features.build import read_feature_schema, read_processed
from shopper_intent.models import registry
from shopper_intent.models.metrics import (
    choose_threshold,
    decile_lift,
    evaluate_predictions,
    policy_summary,
)
from shopper_intent.models.model import IntentModel

MODEL_TYPES = ["logistic", "lightgbm"]
METADATA_FILENAME = "metadata.json"


def build_estimator(config: Settings, model_type: str, seed: int) -> Pipeline:
    """Fresh, unfitted pipeline for one model type."""
    if model_type == "logistic":
        params = config.model.logistic
        classifier = LogisticRegression(
            C=params.C,
            max_iter=params.max_iter,
            class_weight=None if params.class_weight == "none" else params.class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])
    if model_type == "lightgbm":
        params = config.model.lightgbm.model_dump()
        classifier = LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        return Pipeline([("clf", classifier)])
    raise ValueError(f"unknown model type {model_type!r}")


def effective_folds(y: np.ndarray, requested: int) -> int:
    minority = int(min(np.sum(y == 1), np.sum(y == 0)))
    folds = min(requested, minority)
    if folds < 2:
        raise ValueError("need at least two samples of each class for cross-validation")
    return folds


def out_of_fold_probabilities(
    config: Settings,
    model_type: str,
    features: pd.DataFrame,
    target: np.ndarray,
) -> np.ndarray:
    folds = effective_folds(target, config.model.cv_folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.model.seed)
    oof = np.zeros(len(features), dtype=float)
    for train_index, valid_index in splitter.split(features, target):
        estimator = build_estimator(config, model_type, config.model.seed)
        estimator.fit(features.iloc[train_index], target[train_index])
        oof[valid_index] = estimator.predict_proba(features.iloc[valid_index])[:, 1]
    return oof


def fit_calibrator(config: Settings, raw_probabilities: np.ndarray, target: np.ndarray):
    calibrator = IsotonicRegression(
        increasing=True,
        out_of_bounds="clip",
        y_min=config.calibration.clip_low,
        y_max=config.calibration.clip_high,
    )
    calibrator.fit(raw_probabilities, target)
    return calibrator


def _apply_calibrator(config: Settings, calibrator: IsotonicRegression, raw: np.ndarray) -> np.ndarray:
    calibrated = calibrator.predict(raw)
    return np.clip(calibrated, config.calibration.clip_low, config.calibration.clip_high)


def _data_sha256(config: Settings) -> str:
    if config.manifest_path.is_file():
        manifest = json.loads(config.manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256"):
            return str(manifest["sha256"])
    return sha256_file(config.raw_path)


def train_one(
    config: Settings,
    frame: pd.DataFrame,
    model_type: str,
    version: str,
) -> tuple[IntentModel, dict[str, Any], Path]:
    features_schema = read_feature_schema(config)
    feature_names = features_schema["features"]
    target_name = config.features.target

    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    if train_frame.empty or test_frame.empty:
        raise ValueError("processed data has an empty train or test split")

    x_train = train_frame[feature_names]
    y_train = train_frame[target_name].to_numpy().astype(int)
    x_test = test_frame[feature_names]
    y_test = test_frame[target_name].to_numpy().astype(int)

    raw_oof = out_of_fold_probabilities(config, model_type, x_train, y_train)
    calibrator = fit_calibrator(config, raw_oof, y_train)
    calibrated_oof = _apply_calibrator(config, calibrator, raw_oof)
    selection = choose_threshold(
        y_train,
        calibrated_oof,
        config.business.value_per_conversion,
        config.business.cost_per_outreach,
        limit=config.business.threshold_candidates,
    )

    final_estimator = build_estimator(config, model_type, config.model.seed)
    final_estimator.fit(x_train, y_train)
    raw_test = np.clip(final_estimator.predict_proba(x_test)[:, 1], 1e-9, 1 - 1e-9)
    calibrated_test = _apply_calibrator(config, calibrator, raw_test)

    value = config.business.value_per_conversion
    cost = config.business.cost_per_outreach
    threshold = selection["threshold"]
    trained_at = registry.utc_now_iso()
    revision, dirty = registry.git_revision(config.root)

    cv_metrics = evaluate_predictions(y_train, calibrated_oof, threshold, value, cost)
    test_metrics = evaluate_predictions(y_test, calibrated_test, threshold, value, cost)
    cv_policy = policy_summary(
        y_train, calibrated_oof, threshold, config.business.default_threshold, value, cost
    )
    test_policy = policy_summary(
        y_test, calibrated_test, threshold, config.business.default_threshold, value, cost
    )

    model = IntentModel(
        estimator=final_estimator,
        calibrator=calibrator,
        feature_names=feature_names,
        threshold=threshold,
        value_per_conversion=value,
        cost_per_outreach=cost,
        model_type=model_type,
        version=version,
        trained_at=trained_at,
        config_hash=config.config_hash(),
        data_sha256=_data_sha256(config),
        clip_low=config.calibration.clip_low,
        clip_high=config.calibration.clip_high,
    )

    metadata: dict[str, Any] = {
        "version": version,
        "model_type": model_type,
        "seed": config.model.seed,
        "config_hash": config.config_hash(),
        "config_path": str(config.source_path),
        "data_sha256": _data_sha256(config),
        "git_commit": revision,
        "git_dirty": dirty,
        "trained_at": trained_at,
        "threshold": threshold,
        "threshold_selection": {
            "objective": "expected_net_value",
            "source": "out_of_fold_train_probabilities",
            "net_value_per_1000_sessions": selection["net_value_per_1000_sessions"],
            "contact_rate": selection["contact_rate"],
            "candidates": selection["evaluated_thresholds"],
        },
        "business": {
            "value_per_conversion": value,
            "cost_per_outreach": cost,
            "default_threshold": config.business.default_threshold,
        },
        "protocol": {
            "cv_folds": effective_folds(y_train, config.model.cv_folds),
            "cv_source": "stratified k-fold on the training split",
            "calibration": "isotonic, fitted on out-of-fold training probabilities",
            "held_out": "stratified 80/20 split, scored once",
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        },
        "metrics": {
            "cv": cv_metrics,
            "test": test_metrics,
            "cv_policy": cv_policy,
            "test_policy": test_policy,
            "raw_test_brier": float(np.mean((raw_test - y_test) ** 2)),
            "decile_lift_test": decile_lift(y_test, calibrated_test),
        },
        "features": feature_names,
        "oof": {
            "probabilities": "oof_probabilities.npy",
            "labels": "oof_labels.npy",
            "rows": int(len(calibrated_oof)),
            "note": "calibrated out-of-fold training probabilities, aligned with the train split",
        },
    }

    artifact = registry.save(config.models_dir, model, metadata, version=version)
    np.save(artifact / "oof_probabilities.npy", calibrated_oof)
    np.save(artifact / "oof_labels.npy", y_train)
    return model, metadata, artifact


def train(config: Settings, model_types: list[str] | None = None, version: str | None = None) -> dict:
    """Train the requested models and write both artifacts under one version."""
    frame = read_processed(config)
    model_types = model_types or MODEL_TYPES
    version = version or registry.new_version()
    results: dict[str, Any] = {"version": version, "models": {}, "artifacts": {}}
    for model_type in model_types:
        _, metadata, artifact = train_one(config, frame, model_type, version)
        results["models"][model_type] = metadata
        results["artifacts"][model_type] = str(artifact)
    return results
