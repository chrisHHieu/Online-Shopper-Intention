"""Serving layer: health, model info, single and batch prediction, validation."""

from __future__ import annotations

import pytest
from conftest import make_sessions, write_config
from fastapi.testclient import TestClient

from shopper_intent.config import load_config
from shopper_intent.serving.app import create_app


def session_payload(row: int = 0) -> dict:
    frame = make_sessions()
    record = frame.iloc[row].to_dict()
    return {key: value.item() if hasattr(value, "item") else value for key, value in record.items()
            if key != "Revenue"}


def test_health(api_client) -> None:
    response = api_client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_type"] == "lightgbm"


def test_model_info(api_client) -> None:
    response = api_client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["model_type"] == "lightgbm"
    assert 0 < body["threshold"] < 1
    assert body["value_per_conversion"] == 100.0
    assert body["cost_per_outreach"] == 1.0
    assert len(body["feature_names"]) == 32
    assert "pr_auc" in body["metrics"]["test"]
    assert body["protocol"]["cv_folds"] == 5


def test_predict_single(api_client) -> None:
    response = api_client.post("/predict", json=session_payload())
    assert response.status_code == 200, response.text
    body = response.json()
    assert 0.0 <= body["conversion_probability"] <= 1.0
    assert body["decision"] in {"contact", "skip"}
    assert body["threshold"] == pytest.approx(api_client.get("/model/info").json()["threshold"])
    assert len(body["top_contributions"]) == 3
    expected = body["conversion_probability"] * 100.0 - 1.0
    assert body["expected_value_if_contacted"] == pytest.approx(expected, abs=1e-3)
    if body["decision"] == "skip":
        assert body["expected_value_of_decision"] == 0.0


def test_decision_matches_threshold(api_client) -> None:
    body = api_client.post("/predict", json=session_payload(3)).json()
    assert (body["decision"] == "contact") == (body["conversion_probability"] >= body["threshold"])


def test_predict_batch(api_client) -> None:
    payload = {"sessions": [session_payload(index) for index in range(4)]}
    response = api_client.post("/predict/batch", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["predictions"]) == 4
    summary = body["summary"]
    assert summary["sessions"] == 4
    assert summary["contacts"] == sum(
        1 for prediction in body["predictions"] if prediction["decision"] == "contact"
    )
    total = sum(prediction["conversion_probability"] for prediction in body["predictions"])
    assert summary["expected_conversions"] == pytest.approx(total, abs=1e-3)


def test_invalid_month_rejected(api_client) -> None:
    payload = session_payload()
    payload["Month"] = "Jan"
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_unknown_field_rejected(api_client) -> None:
    payload = session_payload()
    payload["SessionId"] = 7
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_negative_count_rejected(api_client) -> None:
    payload = session_payload()
    payload["Administrative"] = -1
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_missing_field_rejected(api_client) -> None:
    payload = session_payload()
    del payload["PageValues"]
    response = api_client.post("/predict", json=payload)
    assert response.status_code == 422


def test_empty_batch_rejected(api_client) -> None:
    response = api_client.post("/predict/batch", json={"sessions": []})
    assert response.status_code == 422


def test_no_artifact_returns_503(tmp_path) -> None:
    config_path = write_config(tmp_path)
    config = load_config(config_path)
    with TestClient(create_app(config)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        response = client.post("/predict", json=session_payload())
        assert response.status_code == 503
