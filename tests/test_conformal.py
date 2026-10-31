"""Split conformal prediction sets: quantile, bands and measured coverage."""

from __future__ import annotations

import numpy as np
import pytest

from shopper_intent.models.conformal import ConformalPolicy, band_report, fit_conformal


class ConformalConfig:
    class conformal:
        enabled = True
        alpha = 0.1


def synthetic(rows: int = 4000, seed: int = 11) -> tuple[np.ndarray, np.ndarray]:
    """Well calibrated probabilities, so conformal coverage should land near 90 percent."""
    rng = np.random.default_rng(seed)
    probability = np.clip(rng.beta(0.8, 3.0, rows), 0.001, 0.999)
    target = (rng.random(rows) < probability).astype(int)
    return target, probability


def test_q_hat_is_a_score_not_a_probability() -> None:
    target, probability = synthetic()
    policy = fit_conformal(ConformalConfig, target, probability)
    assert policy is not None
    assert 0.0 <= policy.q_hat <= 1.0
    assert policy.calibration_rows == len(target)
    assert policy.target_coverage == pytest.approx(0.9)


def test_coverage_holds_on_held_out_rows() -> None:
    target, probability = synthetic(seed=11)
    policy = fit_conformal(ConformalConfig, target[:2000], probability[:2000])
    coverage = policy.coverage(target[2000:], probability[2000:])
    assert policy is not None
    assert 0.85 <= coverage <= 0.95


def test_bands_follow_the_sets() -> None:
    policy = ConformalPolicy(alpha=0.1, q_hat=0.2, calibration_rows=100, target_coverage=0.9)
    probability = np.array([0.01, 0.2, 0.5, 0.81, 0.99])
    assert policy.bands(probability).tolist() == [
        "skip",
        "skip",
        "standard",
        "priority",
        "priority",
    ]
    assert policy.prediction_sets(probability) == [[0], [0], [], [1], [1]]


def test_wide_q_hat_fattens_the_sets() -> None:
    policy = ConformalPolicy(alpha=0.9, q_hat=0.6, calibration_rows=100, target_coverage=0.1)
    sets = policy.prediction_sets(np.array([0.05, 0.5, 0.95]))
    assert sets == [[0], [0, 1], [1]]
    assert "standard" not in policy.bands(np.array([0.05, 0.95])).tolist()


def test_band_report_counts_every_row() -> None:
    target, probability = synthetic()
    policy = fit_conformal(ConformalConfig, target, probability)
    report = band_report(policy, target, probability)
    bands = report["bands"]
    assert set(bands) == {"skip", "standard", "priority"}
    assert sum(row["sessions"] for row in bands.values()) == len(target)
    assert report["mean_set_size"] <= 2.0
    assert 0.85 <= report["coverage"] <= 0.95


def test_disabled_conformal_returns_none() -> None:
    class Disabled:
        class conformal:
            enabled = False
            alpha = 0.1

    target, probability = synthetic(rows=100)
    assert fit_conformal(Disabled, target, probability) is None


def test_disabled_conformal_is_not_an_attribute_error() -> None:
    from conftest import BASE_CONFIG

    from shopper_intent.config import load_config

    config = load_config(BASE_CONFIG)
    assert hasattr(config.conformal, "enabled")
