"""Data contract checks on the synthetic fixture."""

from __future__ import annotations

import pandas as pd
import pytest
from conftest import make_sessions, write_config

from shopper_intent.data.validate import DEFAULT_SCHEMA, validate, validate_frame


def test_fixture_passes(fixture_config, raw_sessions: pd.DataFrame) -> None:
    report = validate_frame(raw_sessions, fixture_config)
    assert report.ok, report.render()
    assert report.row_count == len(raw_sessions)
    assert report.positive_rate is not None
    assert 0.05 < report.positive_rate < 0.5
    # "skip" is the honest status when an optional backend (pandera) is absent.
    assert all(check.status in {"pass", "warn", "skip"} for check in report.checks)


def test_validate_reads_raw_file(fixture_config) -> None:
    report = validate(fixture_config)
    assert report.ok
    assert report.row_count == 120


def test_missing_column_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.drop(columns=["PageValues"])
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any("missing columns" in check.detail for check in report.failures)


def test_negative_count_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken.loc[0, "Administrative"] = -3
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any(check.name == "range:Administrative" for check in report.failures)


def test_rate_above_one_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken.loc[3, "BounceRates"] = 1.4
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any(check.name == "range:BounceRates" for check in report.failures)


def test_unknown_month_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken.loc[5, "Month"] = "Jan"
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any(check.name == "values:Month" for check in report.failures)


def test_null_value_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken.loc[2, "PageValues"] = None
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any(check.name == "nulls:PageValues" for check in report.failures)


def test_non_boolean_target_fails(fixture_config, raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken["Revenue"] = "maybe"
    report = validate_frame(broken, fixture_config)
    assert not report.ok
    assert any(check.name == "target_type" for check in report.failures)


def test_patched_schema_is_used(fixture_config, raw_sessions: pd.DataFrame) -> None:
    schema = dict(DEFAULT_SCHEMA)
    schema["SessionId"] = {"dtype": "int", "min": 0}
    report = validate_frame(raw_sessions, fixture_config, schema=schema)
    assert not report.ok
    assert any("SessionId" in check.detail for check in report.failures)


def test_min_rows_is_enforced(tmp_path, raw_sessions: pd.DataFrame) -> None:
    from shopper_intent.config import load_config

    path = write_config(tmp_path)
    config = load_config(path)
    config.data.min_rows = 10_000
    report = validate_frame(raw_sessions, config)
    assert not report.ok
    assert any(check.name == "row_count" for check in report.failures)


def test_duplicates_above_limit_warn(tmp_path) -> None:
    from shopper_intent.config import load_config

    path = write_config(tmp_path)
    config = load_config(path)
    frame = make_sessions(rows=100)
    doubled = pd.concat([frame, frame.iloc[:20]], ignore_index=True)
    report = validate_frame(doubled, config)
    duplicates = next(check for check in report.checks if check.name == "duplicates")
    assert duplicates.status == "warn"
    assert report.ok


def test_report_renders_readable_text(fixture_config, raw_sessions: pd.DataFrame) -> None:
    text = validate_frame(raw_sessions, fixture_config).render()
    assert "rows=120" in text
    assert "validation passed" in text


def test_missing_raw_file_message(tmp_path) -> None:
    from shopper_intent.config import load_config

    path = write_config(tmp_path)
    config = load_config(path)
    config.raw_path.unlink()
    with pytest.raises(FileNotFoundError, match="fetch"):
        validate(config)
