"""Model factory: availability, imbalance weight and the fitted estimator shape."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from shopper_intent.models import candidates


def test_core_models_are_always_available() -> None:
    available = candidates.available_models()
    assert "logistic" in available
    assert "lightgbm" in available


def test_unknown_model_is_rejected(fixture_config) -> None:
    with pytest.raises(ValueError, match="unknown model type"):
        candidates.build_estimator(fixture_config, "random-forest", 42)


def test_missing_package_names_the_extra(monkeypatch, fixture_config) -> None:
    monkeypatch.setattr(candidates.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(candidates.MissingDependency, match="modern"):
        candidates.require("xgboost")


def test_positive_weight_is_negative_over_positive() -> None:
    target = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
    assert candidates.positive_weight(target) == pytest.approx(7 / 3)


def test_logistic_pipeline_starts_with_a_scaler(fixture_config) -> None:
    estimator = candidates.build_estimator(fixture_config, "logistic", 42)
    assert isinstance(estimator, Pipeline)
    assert [name for name, _ in estimator.steps] == ["scaler", "clf"]


def test_only_tree_models_use_native_categoricals() -> None:
    assert candidates.uses_native_categorical("catboost")
    assert candidates.uses_native_categorical("xgboost")
    assert not candidates.uses_native_categorical("lightgbm")
    assert candidates.categorical_format("xgboost") == "category"


@pytest.mark.modern
def test_xgboost_uses_the_class_ratio(fixture_config) -> None:
    pytest.importorskip("xgboost")
    target = np.array([1] * 10 + [0] * 30)
    estimator = candidates.build_estimator(fixture_config, "xgboost", 42, target)
    classifier = estimator.steps[-1][1]
    params = classifier.get_params()
    assert params["scale_pos_weight"] == pytest.approx(3.0)
    assert params["enable_categorical"] is True
    assert params["tree_method"] == "hist"


@pytest.mark.modern
def test_catboost_declares_its_categorical_columns(fixture_config) -> None:
    pytest.importorskip("catboost")
    estimator = candidates.build_estimator(fixture_config, "catboost", 42)
    classifier = estimator.steps[-1][1]
    assert list(classifier.get_params()["cat_features"]) == list(
        fixture_config.features.categorical
    )
    assert classifier.get_params()["auto_class_weights"] == "Balanced"
    assert classifier.get_params()["eval_metric"] == "PRAUC"
