"""Calibration methods, the out-of-fold choice between them, and ECE."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import yaml
from conftest import BASE_CONFIG

from shopper_intent.config import load_config
from shopper_intent.models.calibration import (
    apply_calibrator,
    fit_calibrator,
    reliability_curve,
)
from shopper_intent.models.metrics import evaluate_predictions, expected_calibration_error


def overconfident_scores(rows: int = 800, seed: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A model whose raw scores are sharper than the truth."""
    rng = np.random.default_rng(seed)
    truth = rng.random(rows) < 0.2
    raw = np.clip(truth * 0.7 + 0.1 + rng.normal(0, 0.08, rows), 0.001, 0.999)
    return raw, truth.astype(int)


def config_with(tmp_path, **calibration):
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["calibration"].update(calibration)
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return load_config(path)


def test_auto_keeps_the_lower_brier_method(tmp_path) -> None:
    config = config_with(tmp_path, method="auto")
    raw, target = overconfident_scores()
    calibrator = fit_calibrator(config, raw, target)
    assert set(calibrator.transforms) == {"isotonic", "platt"}
    chosen = calibrator.selection[f"brier_{calibrator.method}"]
    others = [value for name, value in calibrator.selection.items() if name != f"brier_{calibrator.method}"]
    assert chosen <= min(others)


def test_forced_method_is_respected(tmp_path) -> None:
    raw, target = overconfident_scores()
    platt = fit_calibrator(config_with(tmp_path, method="platt"), raw, target)
    assert platt.method == "platt"
    assert list(platt.transforms) == ["platt"]
    assert platt.requested == "platt"


def test_predictions_are_clipped_and_monotone(tmp_path) -> None:
    config = config_with(tmp_path, method="isotonic", clip_low=0.01, clip_high=0.99)
    raw, target = overconfident_scores()
    calibrator = fit_calibrator(config, raw, target)
    calibrated = calibrator.predict(raw)
    assert calibrated.min() >= 0.01
    assert calibrated.max() <= 0.99
    order = np.argsort(raw)
    assert np.all(np.diff(calibrated[order]) >= -1e-12)


def test_calibration_reduces_brier_on_miscalibrated_scores(tmp_path) -> None:
    from sklearn.metrics import brier_score_loss

    config = config_with(tmp_path, method="auto")
    raw, target = overconfident_scores()
    calibrator = fit_calibrator(config, raw, target)
    calibrated = apply_calibrator(calibrator, raw, 0.0, 1.0)
    assert brier_score_loss(target, calibrated) < brier_score_loss(target, raw)


def test_reliability_curve_is_ordered_and_sized() -> None:
    raw, target = overconfident_scores()
    rows = reliability_curve(target, raw, bins=10)
    assert 2 <= len(rows) <= 10
    lower = [row["bin_lower"] for row in rows]
    assert lower == sorted(lower)
    assert sum(row["count"] for row in rows) <= len(target)


def test_ece_is_zero_for_a_perfect_forecast() -> None:
    rng = np.random.default_rng(5)
    probability = np.full(2000, 0.3)
    target = (rng.random(2000) < 0.3).astype(int)
    assert expected_calibration_error(target, probability) < 0.03


def test_ece_grows_with_distance_from_truth() -> None:
    rng = np.random.default_rng(6)
    target = (rng.random(2000) < 0.3).astype(int)
    good = expected_calibration_error(target, np.full(2000, 0.3), strategy="uniform")
    bad = expected_calibration_error(target, np.full(2000, 0.8), strategy="uniform")
    assert good < 0.05
    assert bad > good


def test_evaluate_predictions_reports_ece() -> None:
    raw, target = overconfident_scores()
    metrics = evaluate_predictions(target, raw, 0.5, 100.0, 1.0)
    assert {"ece", "ece_uniform", "brier", "pr_auc", "roc_auc"} <= set(metrics)
    assert metrics["ece"] >= 0.0


@pytest.mark.parametrize("method", ["isotonic", "platt", "auto"])
def test_every_method_round_trips_through_pickle(tmp_path, method) -> None:
    import pickle

    config = config_with(tmp_path, method=method)
    raw, target = overconfident_scores(rows=200)
    calibrator = fit_calibrator(config, raw, target)
    restored = pickle.loads(pickle.dumps(calibrator))
    np.testing.assert_allclose(restored.predict(raw[:20]), calibrator.predict(raw[:20]))
