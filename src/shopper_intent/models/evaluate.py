"""Evaluation on the held-out split, reports and figures.

``train`` writes one artifact per model with its cross-validated and held-out
metrics. ``evaluate`` reloads the artifacts, rebuilds the comparison report,
computes permutation importance and draws the figures referenced by the README.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_auc_score  # noqa: E402

from shopper_intent.config import Settings  # noqa: E402
from shopper_intent.features.build import read_feature_schema, read_processed  # noqa: E402
from shopper_intent.models import registry  # noqa: E402
from shopper_intent.models.metrics import (  # noqa: E402
    decile_lift,
    evaluate_predictions,
    policy_summary,
    threshold_sweep,
)
from shopper_intent.models.sensitivity import cost_sensitivity  # noqa: E402

PRIMARY_COLORS = {"lightgbm": "#1f5f8b", "logistic": "#c46a1b"}


def _figure_path(config: Settings, name: str) -> Path:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    return config.figures_dir / name


def _save(fig: plt.Figure, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def _primary_model_type(config: Settings) -> str:
    return config.serving.model


def plot_pr_curves(
    config: Settings,
    y_true: np.ndarray,
    probabilities: dict[str, np.ndarray],
    prevalence: float,
) -> Path:
    fig, axis = plt.subplots(figsize=(6.4, 4.6))
    for model_type, proba in probabilities.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        auc = roc_auc_score(y_true, proba)
        axis.plot(
            recall,
            precision,
            label=f"{model_type} (ROC-AUC {auc:.3f})",
            color=PRIMARY_COLORS.get(model_type),
        )
    axis.axhline(prevalence, linestyle="--", color="#777777", label=f"prevalence {prevalence:.3f}")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curve, held-out split")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(loc="upper right", fontsize=8)
    path = _figure_path(config, "pr_curve.png")
    _save(fig, path)
    return path


def plot_calibration(
    config: Settings,
    y_true: np.ndarray,
    raw: np.ndarray,
    calibrated: np.ndarray,
    model_type: str,
) -> Path:
    fig, axis = plt.subplots(figsize=(5.6, 4.8))
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="perfect calibration")
    for label, values, color in (
        ("raw LightGBM scores", raw, "#999999"),
        ("isotonic calibrated", calibrated, PRIMARY_COLORS.get(model_type, "#1f5f8b")),
    ):
        observed, predicted = calibration_curve(y_true, values, n_bins=10, strategy="quantile")
        axis.plot(predicted, observed, marker="o", markersize=3, label=label, color=color)
    axis.set_xlabel("Mean predicted probability")
    axis.set_ylabel("Observed conversion rate")
    axis.set_title("Calibration, held-out split")
    axis.legend(loc="upper left", fontsize=8)
    path = _figure_path(config, "calibration.png")
    _save(fig, path)
    return path


def plot_lift(config: Settings, rows: list[dict[str, Any]]) -> Path:
    fig, axis = plt.subplots(figsize=(6.4, 4.2))
    deciles = [row["decile"] for row in rows]
    lifts = [row["lift"] for row in rows]
    axis.bar([str(decile) for decile in deciles], lifts, color="#1f5f8b")
    axis.axhline(1.0, linestyle="--", color="#777777", label="no lift")
    for index, lift in enumerate(lifts):
        axis.text(index, lift + 0.02, f"{lift:.2f}x", ha="center", fontsize=8)
    axis.set_xlabel("Decile of predicted probability (1 = highest)")
    axis.set_ylabel("Lift over base rate")
    axis.set_title("Lift by decile, held-out split")
    axis.legend(loc="upper right", fontsize=8)
    path = _figure_path(config, "lift_by_decile.png")
    _save(fig, path)
    return path


def plot_threshold_value(
    config: Settings,
    y_true: np.ndarray,
    sweeps: dict[str, list[dict[str, float]]],
    chosen: dict[str, float],
    always_contact_per_1000: float,
) -> Path:
    fig, axis = plt.subplots(figsize=(6.6, 4.6))
    for model_type, sweep in sweeps.items():
        axis.plot(
            [row["threshold"] for row in sweep],
            [row["net_value_per_1000_sessions"] for row in sweep],
            label=model_type,
            color=PRIMARY_COLORS.get(model_type),
        )
    axis.axhline(
        always_contact_per_1000,
        linestyle=":",
        color="#444444",
        label="always contact",
    )
    axis.axhline(0.0, color="#bbbbbb", linewidth=0.8)
    axis.axvline(
        config.business.default_threshold,
        linestyle="--",
        color="#999999",
        linewidth=0.9,
        label="default 0.5",
    )
    for model_type, threshold in chosen.items():
        axis.axvline(
            threshold,
            linestyle="-.",
            color=PRIMARY_COLORS.get(model_type),
            linewidth=1.0,
            label=f"{model_type} chosen {threshold:.3f}",
        )
        nearest = min(
            sweeps[model_type],
            key=lambda row: abs(row["threshold"] - threshold),
        )
        axis.plot(
            nearest["threshold"],
            nearest["net_value_per_1000_sessions"],
            marker="o",
            markersize=6,
            color=PRIMARY_COLORS.get(model_type),
        )
    axis.set_xlabel("Decision threshold")
    axis.set_ylabel("Net value per 1000 sessions")
    axis.set_title("Expected net value against threshold, held-out split")
    axis.legend(loc="lower left", fontsize=8)
    path = _figure_path(config, "threshold_value.png")
    _save(fig, path)
    return path


def plot_importance(config: Settings, rows: list[dict[str, Any]], top: int = 15) -> Path:
    selected = rows[:top][::-1]
    fig, axis = plt.subplots(figsize=(6.8, 5.0))
    axis.barh(
        [row["feature"] for row in selected],
        [row["importance_mean"] for row in selected],
        xerr=[row["importance_std"] for row in selected],
        color="#1f5f8b",
    )
    axis.set_xlabel("Drop in ROC-AUC when the feature is permuted")
    axis.set_title("Permutation importance, held-out split")
    path = _figure_path(config, "permutation_importance.png")
    _save(fig, path)
    return path


def permutation_importance_rows(
    config: Settings,
    model: Any,
    features: pd.DataFrame,
    target: np.ndarray,
    n_repeats: int = 5,
) -> list[dict[str, Any]]:
    """Drop in ROC-AUC when one column is shuffled, averaged over repeats.

    Implemented here rather than with ``sklearn.inspection`` so the artifact
    object does not need a ``fit`` method; the shuffle is seeded and repeatable.
    """
    generator = np.random.default_rng(config.model.seed)
    baseline = float(roc_auc_score(target, model.predict_proba(features)))
    rows: list[dict[str, Any]] = []
    for column in features.columns:
        values = features[column].to_numpy()
        scores = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[column] = generator.permutation(values)
            scores.append(float(roc_auc_score(target, model.predict_proba(shuffled))))
        rows.append(
            {
                "feature": column,
                "importance_mean": float(baseline - np.mean(scores)),
                "importance_std": float(np.std(scores)),
            }
        )
    rows.sort(key=lambda row: -row["importance_mean"])
    return rows


def load_models(config: Settings) -> dict[str, tuple[Any, dict[str, Any]]]:
    loaded: dict[str, tuple[Any, dict[str, Any]]] = {}
    for model_type in ("logistic", "lightgbm"):
        if registry.available_versions(config.models_dir, model_type):
            loaded[model_type] = registry.load_latest(config.models_dir, model_type)
    if not loaded:
        raise FileNotFoundError(
            f"no trained model found under {config.models_dir}; run 'train' first"
        )
    if config.serving.model not in loaded:
        raise FileNotFoundError(
            f"serving.model={config.serving.model!r} has no artifact; train it or change the config"
        )
    return loaded


def evaluate(config: Settings, n_repeats: int = 5) -> dict[str, Any]:
    frame = read_processed(config)
    schema = read_feature_schema(config)
    feature_names = schema["features"]
    target_name = config.features.target

    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    x_test = test_frame[feature_names]
    y_test = test_frame[target_name].to_numpy().astype(int)
    prevalence = float(y_test.mean())
    value = config.business.value_per_conversion
    cost = config.business.cost_per_outreach

    loaded = load_models(config)
    report_models: dict[str, Any] = {}
    probabilities: dict[str, np.ndarray] = {}
    raw_probabilities: dict[str, np.ndarray] = {}
    sweeps: dict[str, list[dict[str, float]]] = {}
    chosen: dict[str, float] = {}

    for model_type, (model, metadata) in loaded.items():
        proba = model.predict_proba(x_test)
        raw = model.predict_proba_raw(x_test)
        probabilities[model_type] = proba
        raw_probabilities[model_type] = raw
        metrics = evaluate_predictions(y_test, proba, model.threshold, value, cost)
        policy = policy_summary(
            y_test, proba, model.threshold, config.business.default_threshold, value, cost
        )
        lift_rows = decile_lift(y_test, proba)
        sweeps[model_type] = threshold_sweep(y_test, proba, value, cost)
        chosen[model_type] = model.threshold
        report_models[model_type] = {
            "version": metadata.get("version"),
            "trained_at": metadata.get("trained_at"),
            "threshold": model.threshold,
            "metrics_cv": metadata.get("metrics", {}).get("cv"),
            "metrics_test": metrics,
            "policy_test": policy,
            "decile_lift_test": lift_rows,
            "brier_raw_test": float(np.mean((raw - y_test) ** 2)),
        }

    primary = _primary_model_type(config)
    primary_model, primary_metadata = loaded[primary]
    importance = permutation_importance_rows(
        config, primary_model, x_test, y_test, n_repeats=n_repeats
    )
    sensitivity = cost_sensitivity(config, model_type=primary)

    pr_path = plot_pr_curves(config, y_test, probabilities, prevalence)
    calibration_path = plot_calibration(
        config,
        y_test,
        raw_probabilities[primary],
        probabilities[primary],
        primary,
    )
    lift_path = plot_lift(config, report_models[primary]["decile_lift_test"])
    threshold_path = plot_threshold_value(
        config,
        y_test,
        sweeps,
        chosen,
        float(np.mean(y_test) * value - cost) * 1000.0,
    )
    importance_path = plot_importance(config, importance)

    lift_csv = config.reports_dir / "decile_lift.csv"
    pd.DataFrame(report_models[primary]["decile_lift_test"]).to_csv(lift_csv, index=False)

    report = {
        "generated_at": registry.utc_now_iso(),
        "primary_model": primary,
        "data": {
            "sha256": primary_metadata.get("data_sha256"),
            "raw_rows": int(schema.get("rows", len(frame))),
            "train_rows": int((frame["split"] == "train").sum()),
            "test_rows": int(len(test_frame)),
            "positive_rate_test": prevalence,
            "positive_rate_train": float(
                frame.loc[frame["split"] == "train", target_name].mean()
            ),
        },
        "business": {
            "value_per_conversion": value,
            "cost_per_outreach": cost,
            "default_threshold": config.business.default_threshold,
        },
        "protocol": {
            "headline": "stratified 5-fold cross-validation on the training split",
            "threshold": "chosen on out-of-fold training probabilities",
            "calibration": "isotonic, fitted on out-of-fold training probabilities",
            "held_out": "stratified 20 percent split, scored once for the numbers below",
        },
        "models": report_models,
        "permutation_importance": importance,
        "cost_sensitivity": {
            "model_type": sensitivity["model_type"],
            "model_version": sensitivity["version"],
            "value_per_conversion": value,
            "costs": [row["cost_per_outreach"] for row in sensitivity["rows"]],
            "rows": sensitivity["rows"],
            "csv": sensitivity["csv"],
            "figure": sensitivity["figure"],
            "note": "threshold per cost chosen on training OOF probabilities, scored on test",
        },
        "figures": [
            pr_path.name,
            calibration_path.name,
            lift_path.name,
            threshold_path.name,
            importance_path.name,
            sensitivity["figure"],
        ],
    }

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    (config.reports_dir / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
