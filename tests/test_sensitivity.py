"""Cost sensitivity sweep on the fixture."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shopper_intent.models import registry
from shopper_intent.models.sensitivity import COLUMNS, cost_sensitivity


def test_one_row_per_cost(trained) -> None:
    config = trained["config"]
    costs = [1.0, 5.0, 10.0, 25.0, 50.0]
    result = cost_sensitivity(config, costs=costs)
    rows = result["rows"]

    assert len(rows) == len(costs)
    assert [row["cost_per_outreach"] for row in rows] == costs
    for row in rows:
        assert list(row) == COLUMNS
        assert 0.0 <= row["threshold"] <= 1.0
        assert 0.0 <= row["contact_rate"] <= 1.0
        assert 0.0 <= row["precision"] <= 1.0
        assert 0.0 <= row["recall"] <= 1.0
        assert row["contacts"] / 30 == pytest.approx(row["contact_rate"])
        assert row["delta_per_1000"] == pytest.approx(
            row["net_value_per_1000_model"] - row["net_value_per_1000_always"]
        )


def test_outputs_written(trained) -> None:
    config = trained["config"]
    cost_sensitivity(config, costs=[1.0, 5.0])
    csv_path = config.reports_dir / "cost_sensitivity.csv"
    figure_path = config.figures_dir / "cost_sensitivity.png"
    assert csv_path.is_file()
    assert figure_path.is_file() and figure_path.stat().st_size > 0
    table = pd.read_csv(csv_path)
    assert list(table.columns) == COLUMNS
    assert len(table) == 2


def test_cost_one_matches_registered_threshold(trained) -> None:
    config = trained["config"]
    model, metadata = registry.load_latest(config.models_dir, config.serving.model)
    result = cost_sensitivity(config, costs=[1.0], model_type=config.serving.model)
    assert result["rows"][0]["threshold"] == pytest.approx(metadata["threshold"])
    assert result["model_type"] == config.serving.model
    assert result["version"] == metadata["version"]


def test_oof_scores_are_persisted(trained) -> None:
    config = trained["config"]
    model, _ = registry.load_latest(config.models_dir, config.serving.model)
    artifact = registry.version_dir(config.models_dir, model.model_type, model.version)
    probabilities = np.load(artifact / "oof_probabilities.npy")
    labels = np.load(artifact / "oof_labels.npy")
    assert probabilities.shape == labels.shape == (90,)
    assert set(np.unique(labels)) == {0, 1}


def test_custom_costs_change_threshold_direction(trained) -> None:
    config = trained["config"]
    cheap = cost_sensitivity(config, costs=[1.0])["rows"][0]
    expensive = cost_sensitivity(config, costs=[100.0])["rows"][0]
    assert expensive["threshold"] >= cheap["threshold"]
