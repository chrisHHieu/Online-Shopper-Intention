"""Training, artifact registry, reproducibility and evaluation report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from shopper_intent.features.build import read_feature_schema, read_processed
from shopper_intent.models import registry
from shopper_intent.models.evaluate import evaluate
from shopper_intent.models.train import train

REQUIRED_METADATA = {
    "version",
    "model_type",
    "seed",
    "config_hash",
    "data_sha256",
    "git_commit",
    "trained_at",
    "threshold",
    "metrics",
    "protocol",
    "business",
}

REQUIRED_METRICS = {
    "precision",
    "recall",
    "f1",
    "pr_auc",
    "roc_auc",
    "brier",
    "ks_at_threshold",
    "confusion_matrix",
    "net_value_per_1000_sessions",
}


def test_artifacts_written(trained) -> None:
    results = trained["results"]
    assert results["version"] == "test-0001"
    assert set(results["models"]) == {"logistic", "lightgbm"}
    for model_type, path in results["artifacts"].items():
        directory = registry.version_dir(trained["config"].models_dir, model_type, "test-0001")
        assert str(directory) == path
        assert (directory / "model.joblib").is_file()
        assert (directory / "metadata.json").is_file()
        pointer = registry.model_dir(trained["config"].models_dir, model_type) / "latest"
        assert pointer.read_text(encoding="utf-8").strip() == "test-0001"


def test_metadata_contract(trained) -> None:
    config = trained["config"]
    for model_type, metadata in trained["results"]["models"].items():
        assert REQUIRED_METADATA.issubset(metadata)
        assert metadata["model_type"] == model_type
        assert metadata["seed"] == config.model.seed
        assert 0.0 < metadata["threshold"] < 1.0
        assert REQUIRED_METRICS.issubset(metadata["metrics"]["cv"])
        assert REQUIRED_METRICS.issubset(metadata["metrics"]["test"])
        assert metadata["metrics"]["cv"]["confusion_matrix"].keys() == {"tn", "fp", "fn", "tp"}
        assert metadata["protocol"]["cv_folds"] == 5
        assert metadata["protocol"]["train_rows"] + metadata["protocol"]["test_rows"] == 120
        assert len(metadata["data_sha256"]) == 64


def test_training_is_reproducible(trained) -> None:
    config = trained["config"]
    first = trained["results"]["models"]
    second = train(config, version="test-0002")["models"]
    for model_type in ("logistic", "lightgbm"):
        assert second[model_type]["threshold"] == first[model_type]["threshold"]
        for metric in ("pr_auc", "roc_auc", "brier", "f1"):
            assert (
                second[model_type]["metrics"]["cv"][metric]
                == first[model_type]["metrics"]["cv"][metric]
            )
            assert (
                second[model_type]["metrics"]["test"][metric]
                == first[model_type]["metrics"]["test"][metric]
            )


def test_saved_model_reloads_and_predicts(trained) -> None:
    config = trained["config"]
    frame = read_processed(config)
    schema = read_feature_schema(config)
    test_frame = frame[frame["split"] == "test"].reset_index(drop=True)
    model, metadata = registry.load_latest(config.models_dir, "lightgbm")

    probability = model.predict_proba(test_frame[schema["features"]])
    assert probability.shape == (len(test_frame),)
    assert np.all((probability >= 0) & (probability <= 1))
    assert model.threshold == metadata["threshold"]

    prediction = model.predict(test_frame[schema["features"]])
    assert set(np.unique(prediction["decision"])).issubset({"contact", "skip"})
    expected = probability * model.value_per_conversion - model.cost_per_outreach
    np.testing.assert_allclose(prediction["expected_value_if_contacted"], expected)
    assert np.all(prediction["expected_value_of_decision"] >= 0)


def test_explanations_are_ranked(trained) -> None:
    config = trained["config"]
    frame = read_processed(config)
    schema = read_feature_schema(config)
    rows = frame[frame["split"] == "test"].head(5)
    model, _ = registry.load_latest(config.models_dir, "lightgbm")
    explanations = model.explain(rows[schema["features"]], top_k=3)
    assert len(explanations) == 5
    for row in explanations:
        assert len(row) == 3
        magnitudes = [abs(item["contribution"]) for item in row]
        assert magnitudes == sorted(magnitudes, reverse=True)
        assert all(item["direction"] in {"increases", "decreases"} for item in row)


def test_logistic_contributions_exist(trained) -> None:
    config = trained["config"]
    frame = read_processed(config)
    schema = read_feature_schema(config)
    model, _ = registry.load_latest(config.models_dir, "logistic")
    explanations = model.explain(frame[schema["features"]].head(2))
    assert len(explanations) == 2
    assert len(explanations[0]) == 3


@pytest.fixture(scope="module")
def evaluation(trained):
    return evaluate(trained["config"], n_repeats=2)


def test_evaluate_writes_report_and_figures(trained, evaluation) -> None:
    config = trained["config"]
    report = evaluation
    assert report["primary_model"] == config.serving.model
    assert set(report["models"]) == {"logistic", "lightgbm"}
    assert len(report["models"]["lightgbm"]["decile_lift_test"]) == 10
    assert len(report["permutation_importance"]) == 32
    assert report["data"]["test_rows"] == 30

    report_path = config.reports_dir / "metrics.json"
    assert report_path.is_file()
    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved["primary_model"] == report["primary_model"]
    for figure in report["figures"]:
        assert (config.figures_dir / figure).is_file()
    lift = pd.read_csv(config.reports_dir / "decile_lift.csv")
    assert len(lift) == 10


def test_decile_lift_table_arithmetic(trained, evaluation) -> None:
    config = trained["config"]
    report = evaluation
    rows = report["models"]["lightgbm"]["decile_lift_test"]
    frame = read_processed(config)
    test_frame = frame[frame["split"] == "test"]
    prevalence = float(test_frame["Revenue"].mean())

    assert [row["decile"] for row in rows] == list(range(1, 11))
    assert sum(row["sessions"] for row in rows) == len(test_frame)
    assert sum(row["positives"] for row in rows) == int(test_frame["Revenue"].sum())
    for row in rows:
        assert row["response_rate"] == row["positives"] / row["sessions"]
        assert row["lift"] == pytest.approx(row["response_rate"] / prevalence)
    captures = [row["cumulative_capture"] for row in rows]
    assert captures == sorted(captures)
    assert captures[-1] == 1.0
    scores = [row["max_score"] for row in rows]
    assert scores == sorted(scores, reverse=True)
