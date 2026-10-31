"""Training behaviour that needs the modern extras: champion, conformal, native cats."""

from __future__ import annotations

import json

import numpy as np
import pytest

from shopper_intent.features.build import read_feature_schema, read_processed
from shopper_intent.models import registry
from shopper_intent.models.train import train

pytestmark = pytest.mark.modern


def test_champion_is_the_best_out_of_fold_value(trained) -> None:
    results = trained["results"]
    values = {
        model_type: metadata["metrics"]["cv"]["net_value_per_1000_sessions"]
        for model_type, metadata in results["models"].items()
    }
    assert results["champion"] == max(values, key=values.get)
    assert "out-of-fold" in results["selection_rule"]


def test_every_model_carries_a_conformal_policy(trained) -> None:
    config = trained["config"]
    for model_type in trained["results"]["models"]:
        model, metadata = registry.load_latest(config.models_dir, model_type)
        assert model.conformal is not None
        assert 0.0 < model.conformal.q_hat < 1.0
        assert metadata["metrics"]["conformal_test"]["coverage"] is not None


def test_coverage_is_measured_on_the_held_out_split(trained) -> None:
    config = trained["config"]
    frame = read_processed(config)
    schema = read_feature_schema(config)
    test = frame[frame["split"] == "test"]
    model, metadata = registry.load_latest(config.models_dir, "logistic")
    probability = model.predict_proba(test[schema["features"]])
    coverage = model.coverage(test["Revenue"].to_numpy(), probability)
    assert coverage == pytest.approx(metadata["metrics"]["conformal_test"]["coverage"])
    assert 0.0 <= coverage <= 1.0


def test_bands_partition_the_rows(trained) -> None:
    config = trained["config"]
    frame = read_processed(config)
    schema = read_feature_schema(config)
    test = frame[frame["split"] == "test"]
    model, _ = registry.load_latest(config.models_dir, "logistic")
    probability = model.predict_proba(test[schema["features"]])
    bands = model.band(probability)
    assert set(bands) <= {"skip", "standard", "priority"}
    assert len(bands) == len(test)


def test_catalog_model_can_be_trained_and_served(fixture_config, tmp_path) -> None:
    pytest.importorskip("catboost")
    config = fixture_config.model_copy(
        update={
            "model": fixture_config.model.model_copy(
                update={
                    "candidates": ["logistic", "catboost"],
                    "catboost": fixture_config.model.catboost.model_copy(
                        update={"iterations": 60, "depth": 3}
                    ),
                }
            )
        }
    )
    results = train(config, model_types=["catboost"], version="cat-0001")
    metadata = results["models"]["catboost"]
    assert metadata["protocol"]["native_categorical"] is True
    model, _ = registry.load_latest(config.models_dir, "catboost")
    frame = read_processed(config)
    schema = read_feature_schema(config)
    test = frame[frame["split"] == "test"]
    probability = model.predict_proba(test[schema["features"]])
    assert np.isfinite(probability).all()
    reasons = model.explain(test[schema["features"]].head(2), top_k=2)
    assert all("reason" in row for row in reasons[0])
    saved = json.loads((config.models_dir / "catboost" / "cat-0001" / "metadata.json").read_text())
    assert saved["calibration"]["method"] in {"isotonic", "platt"}
