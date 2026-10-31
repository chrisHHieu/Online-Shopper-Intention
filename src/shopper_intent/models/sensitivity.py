"""Cost sensitivity: where the model earns its keep as outreach gets expensive.

For each cost per outreach the threshold is chosen on the training out-of-fold
probabilities only. The held-out split is then scored once per cost and compared
with the always-contact policy, which pays the same cost for every session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shopper_intent.config import Settings  # noqa: E402
from shopper_intent.features.build import read_feature_schema, read_processed  # noqa: E402
from shopper_intent.models import registry  # noqa: E402
from shopper_intent.models.metrics import (  # noqa: E402
    choose_threshold,
    evaluate_predictions,
    value_summary,
)
from shopper_intent.models.train import out_of_fold_probabilities  # noqa: E402

CSV_NAME = "cost_sensitivity.csv"
FIGURE_NAME = "cost_sensitivity.png"

COLUMNS = [
    "cost_per_outreach",
    "threshold",
    "contact_rate",
    "contacts",
    "precision",
    "recall",
    "net_value_per_1000_model",
    "net_value_per_1000_always",
    "delta_per_1000",
]


def _calibrate(model: Any, raw: np.ndarray) -> np.ndarray:
    if model.calibrator is None:
        return np.asarray(raw, dtype=float)
    calibrated = np.asarray(model.calibrator.predict(raw), dtype=float)
    return np.clip(calibrated, model.clip_low, model.clip_high)


def training_oof(config: Settings, model: Any, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Calibrated out-of-fold probabilities from training, saved with the artifact."""
    artifact = registry.version_dir(config.models_dir, model.model_type, model.version)
    probabilities_path = artifact / "oof_probabilities.npy"
    labels_path = artifact / "oof_labels.npy"
    if probabilities_path.is_file() and labels_path.is_file():
        probabilities = np.load(probabilities_path)
        labels = np.load(labels_path).astype(int)
        if len(probabilities) == len(labels):
            return probabilities, labels

    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    schema = read_feature_schema(config)
    features = train_frame[schema["features"]]
    labels = train_frame[config.features.target].to_numpy().astype(int)
    raw = out_of_fold_probabilities(config, model.model_type, features, labels)
    return _calibrate(model, raw), labels


def _plot(config: Settings, table: pd.DataFrame) -> Path:
    config.figures_dir.mkdir(parents=True, exist_ok=True)
    path = config.figures_dir / FIGURE_NAME
    costs = table["cost_per_outreach"].to_numpy()
    figure, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.4))

    left.plot(costs, table["net_value_per_1000_model"], marker="o", color="#1f5f8b", label="model, tuned threshold")
    left.plot(
        costs,
        table["net_value_per_1000_always"],
        marker="s",
        linestyle="--",
        color="#c0522a",
        label="always contact",
    )
    left.axhline(0.0, color="#666666", linewidth=0.8)
    left.set_xlabel("Cost per outreach (value per conversion 100)")
    left.set_ylabel("Net value per 1000 sessions")
    left.set_title("Net value against outreach cost")
    left.legend(fontsize=8)

    right.bar([f"{cost:g}" for cost in costs], table["delta_per_1000"], color="#1f5f8b")
    right.axhline(0.0, color="#666666", linewidth=0.8)
    right.set_xlabel("Cost per outreach")
    right.set_ylabel("Model minus always contact")
    right.set_title("Gain over always contact")

    figure.suptitle("Cost sensitivity, held-out split", fontsize=10)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)
    return path


def cost_sensitivity(
    config: Settings,
    costs: list[float] | None = None,
    model_type: str | None = None,
) -> dict[str, Any]:
    """Sweep outreach cost, choose each threshold on train OOF, score test once per cost."""
    model_type = model_type or config.serving.model
    model, metadata = registry.load_latest(config.models_dir, model_type)
    frame = read_processed(config)
    schema = read_feature_schema(config)

    probabilities, labels = training_oof(config, model, frame)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    x_test = test_frame[schema["features"]]
    y_test = test_frame[config.features.target].to_numpy().astype(int)
    test_probability = model.predict_proba(x_test)

    value = config.business.value_per_conversion
    requested = [float(cost) for cost in (costs or config.business.sensitivity_costs)]
    rows: list[dict[str, float]] = []
    for cost in requested:
        selection = choose_threshold(
            labels,
            probabilities,
            value,
            cost,
            limit=config.business.threshold_candidates,
        )
        threshold = float(selection["threshold"])
        metrics = evaluate_predictions(y_test, test_probability, threshold, value, cost)
        always = value_summary(y_test, np.ones_like(test_probability), 0.0, value, cost)
        rows.append(
            {
                "cost_per_outreach": cost,
                "threshold": threshold,
                "contact_rate": float(metrics["contact_rate"]),
                "contacts": int(metrics["contacted"]),
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "net_value_per_1000_model": float(metrics["net_value_per_1000_sessions"]),
                "net_value_per_1000_always": float(always["net_value_per_1000_sessions"]),
                "delta_per_1000": float(
                    metrics["net_value_per_1000_sessions"]
                    - always["net_value_per_1000_sessions"]
                ),
            }
        )

    table = pd.DataFrame(rows, columns=COLUMNS)
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = config.reports_dir / CSV_NAME
    table.to_csv(csv_path, index=False)
    figure_path = _plot(config, table)

    return {
        "model_type": model_type,
        "version": metadata["version"],
        "value_per_conversion": value,
        "rows": rows,
        "csv": CSV_NAME,
        "figure": FIGURE_NAME,
        "csv_path": str(csv_path),
        "figure_path": str(figure_path),
    }
