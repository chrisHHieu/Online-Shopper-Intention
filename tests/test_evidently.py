"""Evidently drift report on the fixture, split by month."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shopper_intent.monitor.evidently_report import (
    EvidentlyUnavailable,
    drift_evidently,
    month_periods,
)

pytestmark = pytest.mark.modern


def test_month_periods_split_at_september(raw_sessions) -> None:
    reference, current = month_periods(raw_sessions)
    assert "Sep" not in reference
    assert "Sep" in current
    assert set(reference).isdisjoint(current)


def test_report_writes_html_and_json(fixture_config) -> None:
    pytest.importorskip("evidently")
    payload = drift_evidently(
        fixture_config,
        reference_months=["Feb", "Mar", "May"],
        current_months=["Sep", "Oct", "Nov", "Dec"],
    )
    html = Path(payload["html"])
    assert html.is_file()
    assert html.stat().st_size > 1000
    assert payload["features"]["columns_checked"] > 0
    assert payload["rows"]["reference"] > 0
    assert payload["rows"]["current"] > 0
    saved = json.loads(
        (fixture_config.reports_dir / "evidently_drift.json").read_text(encoding="utf-8")
    )
    assert saved["engine"] == "evidently"
    assert saved["features"]["drifted_count"] >= 0


def test_target_drift_is_reported_next_to_features(fixture_config) -> None:
    pytest.importorskip("evidently")
    payload = drift_evidently(
        fixture_config,
        reference_months=["Feb", "Mar", "May"],
        current_months=["Sep", "Oct", "Nov"],
    )
    target = payload["target"]
    assert 0.0 <= target["reference_rate"] <= 1.0
    assert 0.0 <= target["current_rate"] <= 1.0
    assert target["delta"] == pytest.approx(target["current_rate"] - target["reference_rate"])
    assert target["psi"] >= 0.0


def test_empty_period_is_refused(fixture_config) -> None:
    pytest.importorskip("evidently")
    with pytest.raises(ValueError, match="empty"):
        drift_evidently(fixture_config, reference_months=["Feb"], current_months=["Apr"])


def test_missing_evidently_is_reported(monkeypatch, fixture_config) -> None:
    monkeypatch.setattr("shopper_intent.monitor.evidently_report.EVIDENTLY_AVAILABLE", False)
    with pytest.raises(EvidentlyUnavailable, match="modern"):
        drift_evidently(fixture_config)
