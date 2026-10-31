"""Training entry point.

Protocol: one stratified 80/20 split. Inside the training split, stratified
k-fold cross-validation produces out-of-fold probabilities. Those are used for
every headline number, for choosing and fitting the probability calibrator, for
fitting the conformal quantile, and for choosing the business threshold. The
held-out split is scored once and only reports, including the conformal coverage
that is reported but never used to fit anything.

The calibrator and the conformal quantile are fitted on out-of-fold predictions
from models trained on 4/5 of the training split and then applied to a model
refitted on all of it. This is the usual compromise; the held-out split is where
calibration and coverage quality are judged.

When several candidates are trained in one call, the champion is the one with
the highest out-of-fold net value per 1000 sessions. The held-out split never
decides which model ships.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from shopper_intent.config import Settings
from shopper_intent.data.fetch import sha256_file
from shopper_intent.features.build import read_feature_schema, read_processed
from shopper_intent.models import registry
from shopper_intent.models.calibration import apply_calibrator, fit_calibrator, reliability_curve
from shopper_intent.models.candidates import build_estimator, require, uses_native_categorical
from shopper_intent.models.conformal import band_report, fit_conformal
from shopper_intent.models.metrics import (
    choose_threshold,
    decile_lift,
    evaluate_predictions,
    policy_summary,
)
from shopper_intent.models.model import IntentModel

MODEL_TYPES = ["logistic", "lightgbm", "xgboost", "catboost"]
METADATA_FILENAME = "metadata.json"


def effective_folds(y: np.ndarray, requested: int) -> int:
    minority = int(min(np.sum(y == 1), np.sum(y == 0)))
    folds = min(requested, minority)
    if folds < 2:
        raise ValueError("need at least two samples of each class for cross-validation")
    return folds


def fold_indices(
    config: Settings,
    features: pd.DataFrame,
    target: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified folds, shared by training and tuning."""
    folds = effective_folds(target, config.model.cv_folds)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=config.model.seed)
    return list(splitter.split(features, target))


def out_of_fold_probabilities(
    config: Settings,
    model_type: str,
    features: pd.DataFrame,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]] | None = None,
    prepare: Any = None,
) -> np.ndarray:
    """Out-of-fold raw scores; each fold's estimator never sees its validation rows."""
    folds = folds or fold_indices(config, features, target)
    oof = np.zeros(len(features), dtype=float)
    for train_index, valid_index in folds:
        estimator = build_estimator(config, model_type, config.model.seed, target[train_index])
        train_matrix = prepare(features.iloc[train_index]) if prepare else features.iloc[train_index]
        valid_matrix = prepare(features.iloc[valid_index]) if prepare else features.iloc[valid_index]
        estimator.fit(train_matrix, target[train_index])
        oof[valid_index] = estimator.predict_proba(valid_matrix)[:, 1]
    return oof


def _native_prepare(config: Settings, model_type: str):
    """Conversion from the one-hot contract to the native table, or None."""
    if not uses_native_categorical(model_type):
        return None
    from shopper_intent.features.build import to_native

    vocabulary = config.features.categorical

    def prepare(matrix: pd.DataFrame) -> pd.DataFrame:
        return to_native(matrix, vocabulary, as_category=model_type == "xgboost")

    return prepare


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
    require(model_type)
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

    folds = fold_indices(config, x_train, y_train)
    prepare = _native_prepare(config, model_type)
    raw_oof = out_of_fold_probabilities(config, model_type, x_train, y_train, folds, prepare)
    calibrator = fit_calibrator(config, raw_oof, y_train)
    calibrated_oof = apply_calibrator(
        calibrator, raw_oof, config.calibration.clip_low, config.calibration.clip_high
    )
    conformal = fit_conformal(config, y_train, calibrated_oof)

    value = config.business.value_per_conversion
    cost = config.business.cost_per_outreach
    selection = choose_threshold(
        y_train,
        calibrated_oof,
        value,
        cost,
        limit=config.business.threshold_candidates,
    )

    final_estimator = build_estimator(config, model_type, config.model.seed, y_train)
    train_matrix = prepare(x_train) if prepare else x_train
    final_estimator.fit(train_matrix, y_train)
    raw_test = np.clip(final_estimator.predict_proba(x_test if prepare is None else prepare(x_test))[:, 1], 1e-9, 1 - 1e-9)
    calibrated_test = apply_calibrator(
        calibrator, raw_test, config.calibration.clip_low, config.calibration.clip_high
    )

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
        categorical_vocabulary=config.features.categorical,
        native_categorical=uses_native_categorical(model_type),
        conformal=conformal,
    )

    conformal_block: dict[str, Any] | None = None
    if conformal is not None:
        conformal_block = {
            **band_report(conformal, y_test, calibrated_test),
            "calibration_source": "out-of-fold training probabilities",
            "test_coverage": float(conformal.coverage(y_test, calibrated_test)),
            "test_alpha": float(config.conformal.alpha),
        }

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
            "cv_folds": len(folds),
            "cv_source": "stratified k-fold on the training split",
            "calibration": (
                f"{calibrator.method}, fitted on out-of-fold training probabilities"
            ),
            "held_out": "stratified 80/20 split, scored once",
            "conformal": (
                f"split conformal, alpha {conformal.alpha}, quantile fitted on out-of-fold "
                "training probabilities"
                if conformal is not None
                else "disabled"
            ),
            "native_categorical": uses_native_categorical(model_type),
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "positive_rate_train": float(y_train.mean()),
            "positive_rate_test": float(y_test.mean()),
        },
        "calibration": {
            **calibrator.info(),
            "oof_reliability": reliability_curve(y_train, calibrated_oof),
            "raw_oof_brier": float(np.mean((raw_oof - y_train) ** 2)),
            "calibrated_oof_brier": float(np.mean((calibrated_oof - y_train) ** 2)),
        },
        "conformal": conformal.info() if conformal is not None else None,
        "metrics": {
            "cv": cv_metrics,
            "test": test_metrics,
            "cv_policy": cv_policy,
            "test_policy": test_policy,
            "raw_test_brier": float(np.mean((raw_test - y_test) ** 2)),
            "decile_lift_test": decile_lift(y_test, calibrated_test),
            "conformal_test": conformal_block,
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


def champion_of(results: dict[str, Any]) -> str | None:
    """Model with the highest out-of-fold net value per 1000 sessions."""
    scored = [
        (metadata["metrics"]["cv"]["net_value_per_1000_sessions"], model_type)
        for model_type, metadata in results["models"].items()
    ]
    if not scored:
        return None
    return max(scored)[1]


def train(
    config: Settings,
    model_types: list[str] | None = None,
    version: str | None = None,
    track: bool | None = None,
) -> dict:
    """Train the requested models and write their artifacts under one version."""
    frame = read_processed(config)
    model_types = list(model_types or config.model.candidates)
    version = version or registry.new_version()
    results: dict[str, Any] = {"version": version, "models": {}, "artifacts": {}}
    for model_type in model_types:
        _, metadata, artifact = train_one(config, frame, model_type, version)
        results["models"][model_type] = metadata
        results["artifacts"][model_type] = str(artifact)
    results["champion"] = champion_of(results)
    results["selection_rule"] = "highest out-of-fold net value per 1000 sessions"

    should_track = config.tracking.enabled if track is None else track
    if should_track:
        from shopper_intent.tracking.mlflow import log_training_run

        results["tracking"] = log_training_run(config, results)
    return results
