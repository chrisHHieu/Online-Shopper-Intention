"""PSI arithmetic and the drift report on the fixture."""

from __future__ import annotations

import json

import numpy as np

from shopper_intent.monitor.drift import drift, psi, status_for


def test_identical_distribution_has_zero_psi() -> None:
    values = np.linspace(0, 1, 200)
    assert psi(values, values.copy()) == 0.0


def test_constant_reference_is_guarded() -> None:
    assert psi(np.zeros(50), np.zeros(30)) == 0.0
    assert psi(np.zeros(50), np.ones(30)) == 10.0


def test_shifted_distribution_is_significant() -> None:
    reference = np.random.default_rng(1).normal(0, 1, 2000)
    shifted = np.random.default_rng(2).normal(3, 1, 2000)
    assert psi(reference, shifted) > 0.25


def test_status_bands() -> None:
    assert status_for(0.05) == "stable"
    assert status_for(0.15) == "moderate"
    assert status_for(0.5) == "significant"


def test_drift_report_written(trained) -> None:
    config = trained["config"]
    report = drift(config)
    assert report["reference_rows"] == 90
    assert report["current_rows"] == 30
    assert report["summary"]["features_checked"] == 32
    significant = [row for row in report["features"] if row["status"] == "significant"]
    assert report["summary"]["drift_detected"] == bool(significant)
    assert report["current"] == "held_out_test_split"
    assert len(report["features"]) == 32
    saved = json.loads((config.reports_dir / "drift.json").read_text(encoding="utf-8"))
    assert saved["summary"] == report["summary"]
    psi_values = [row["psi"] for row in report["features"]]
    assert psi_values == sorted(psi_values, reverse=True)
