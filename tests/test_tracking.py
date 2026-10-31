"""MLflow tracking and registration, and the promise that it never blocks training."""

from __future__ import annotations

import sys

import pytest

from shopper_intent.tracking.mlflow import log_training_run

pytestmark = pytest.mark.modern


def test_run_is_logged_and_models_registered(trained) -> None:
    pytest.importorskip("mlflow")
    outcome = log_training_run(trained["config"], trained["results"])
    assert outcome["enabled"] is True
    assert outcome.get("fallback") is None, outcome
    assert outcome["run_id"]
    assert set(outcome["models"]) == set(trained["results"]["models"])
    for model_type, entry in outcome["models"].items():
        assert entry["registered_model"] == f"shopper_intent_{model_type}"
        assert str(entry["version"]).isdigit()
        assert "error" not in entry


def test_champion_gets_the_alias(trained) -> None:
    pytest.importorskip("mlflow")
    outcome = log_training_run(trained["config"], trained["results"])
    champion = trained["results"]["champion"]
    assert champion in outcome["models"]
    assert outcome["models"][champion].get("alias") == "champion"


def test_registered_model_can_be_loaded_back(trained) -> None:
    mlflow = pytest.importorskip("mlflow")
    outcome = log_training_run(trained["config"], trained["results"])
    champion = trained["results"]["champion"]
    entry = outcome["models"][champion]
    mlflow.set_tracking_uri(outcome["tracking_uri"])
    model = mlflow.pyfunc.load_model(
        f"models:/{entry['registered_model']}/{entry['version']}"
    )
    assert model is not None


def test_tracking_degrades_when_mlflow_is_missing(monkeypatch, trained) -> None:
    monkeypatch.setitem(sys.modules, "mlflow", None)
    outcome = log_training_run(trained["config"], trained["results"])
    assert outcome["enabled"] is False
    assert "mlflow unavailable" in outcome["reason"]
