"""Feature build: schema, ratio arithmetic, determinism, split."""

from __future__ import annotations

import numpy as np
import pandas as pd
from conftest import make_sessions

from shopper_intent.features.build import (
    RAW_COLUMNS,
    build_features,
    build_training_frame,
    feature_columns,
    prepare_for_model,
    read_feature_schema,
    read_processed,
    write_processed,
)


def test_feature_columns_are_stable(fixture_config) -> None:
    columns = feature_columns(fixture_config)
    assert columns == feature_columns(fixture_config)
    assert len(columns) == 32
    assert columns[:14] == fixture_config.features.numeric
    assert columns[14:17] == ["PageValuePerProductPage", "DurationPerPage", "BounceExitSpread"]
    assert "Month_May" in columns
    assert "VisitorType_New_Visitor" in columns


def test_no_target_in_features(fixture_config, raw_sessions: pd.DataFrame) -> None:
    matrix = build_features(raw_sessions, fixture_config)
    assert "Revenue" not in matrix.columns
    assert "split" not in matrix.columns
    assert matrix.isna().sum().sum() == 0
    assert np.isfinite(matrix.to_numpy()).all()


def test_ratio_arithmetic(fixture_config, raw_sessions: pd.DataFrame) -> None:
    row = raw_sessions.iloc[0]
    matrix = build_features(raw_sessions, fixture_config)
    pages = row["Administrative"] + row["Informational"] + row["ProductRelated"]
    duration = (
        row["Administrative_Duration"]
        + row["Informational_Duration"]
        + row["ProductRelated_Duration"]
    )
    assert matrix["PageValuePerProductPage"].iloc[0] == pytest_approx(
        row["PageValues"] / (row["ProductRelated"] + 1.0)
    )
    assert matrix["DurationPerPage"].iloc[0] == pytest_approx(duration / (pages + 1.0))
    assert matrix["BounceExitSpread"].iloc[0] == pytest_approx(
        row["ExitRates"] - row["BounceRates"]
    )


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, rel=1e-9, abs=1e-9)


def test_build_is_deterministic(fixture_config, raw_sessions: pd.DataFrame) -> None:
    first = build_features(raw_sessions, fixture_config)
    second = build_features(raw_sessions.copy(), fixture_config)
    pd.testing.assert_frame_equal(first, second)


def test_split_sizes_and_stratification(fixture_config, raw_sessions: pd.DataFrame) -> None:
    frame = build_training_frame(raw_sessions, fixture_config)
    counts = frame["split"].value_counts()
    assert counts["test"] == 30
    assert counts["train"] == 90
    overall = frame["Revenue"].mean()
    train_rate = frame.loc[frame["split"] == "train", "Revenue"].mean()
    test_rate = frame.loc[frame["split"] == "test", "Revenue"].mean()
    assert abs(train_rate - overall) < 0.05
    assert abs(test_rate - overall) < 0.08


def test_split_is_reproducible(fixture_config, raw_sessions: pd.DataFrame) -> None:
    first = build_training_frame(raw_sessions, fixture_config)
    second = build_training_frame(raw_sessions, fixture_config)
    pd.testing.assert_series_equal(first["split"], second["split"])


def test_processed_roundtrip(fixture_config) -> None:
    path = write_processed(fixture_config)
    assert path.is_file()
    frame = read_processed(fixture_config)
    schema = read_feature_schema(fixture_config)
    assert schema["rows"] == 120
    assert schema["train_rows"] + schema["test_rows"] == 120
    assert all(column in frame.columns for column in schema["features"])


def test_prepare_for_model_matches_training_order(fixture_config) -> None:
    schema = read_feature_schema(fixture_config)
    matrix = prepare_for_model(make_sessions(rows=3), fixture_config)
    assert list(matrix.columns) == schema["features"]


def test_missing_raw_column_raises(fixture_config, raw_sessions: pd.DataFrame) -> None:
    import pytest

    broken = raw_sessions.drop(columns=["VisitorType"])
    with pytest.raises(ValueError, match="missing columns"):
        build_features(broken, fixture_config)


def test_raw_column_list_matches_source() -> None:
    assert len(RAW_COLUMNS) == 17
    assert RAW_COLUMNS[0] == "Administrative"
    assert RAW_COLUMNS[-1] == "Weekend"
