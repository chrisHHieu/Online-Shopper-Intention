"""Pandera contracts layered on the declarative data contract."""

from __future__ import annotations

import pandas as pd
import pytest

from shopper_intent.data.contracts import PANDERA_AVAILABLE, check_raw
from shopper_intent.data.validate import validate_frame, validate_processed

pytestmark = pytest.mark.modern


def test_pandera_is_installed_for_the_modern_job() -> None:
    assert PANDERA_AVAILABLE is True


def test_raw_contract_passes_on_the_fixture(raw_sessions: pd.DataFrame) -> None:
    checks = check_raw(raw_sessions)
    assert [check.status for check in checks] == ["pass"], checks


def test_raw_contract_catches_a_non_numeric_count(raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken["Administrative"] = broken["Administrative"].astype(object)
    broken.loc[0, "Administrative"] = "twelve"
    checks = check_raw(broken)
    assert any(check.status == "fail" for check in checks)


def test_raw_contract_catches_an_out_of_range_rate(raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken.loc[0, "BounceRates"] = 1.5
    checks = check_raw(broken)
    assert any(check.status == "fail" for check in checks)


def test_raw_contract_catches_an_unexpected_column(raw_sessions: pd.DataFrame) -> None:
    broken = raw_sessions.copy()
    broken["NotAColumn"] = 1.0
    report = validate_frame(broken, _config(), pandera=True)
    assert not report.ok


def test_validate_reports_both_layers(fixture_config, raw_sessions: pd.DataFrame) -> None:
    report = validate_frame(raw_sessions, fixture_config)
    names = [check.name for check in report.checks]
    assert any(name.startswith("pandera:") for name in names)
    assert "target_rate" in names


def test_processed_contract_accepts_the_built_table(fixture_config) -> None:
    from shopper_intent.features.build import write_processed

    write_processed(fixture_config)
    report = validate_processed(fixture_config)
    assert report.ok, report.render()


def test_processed_contract_rejects_a_bad_label(fixture_config) -> None:
    from shopper_intent.features.build import read_feature_schema, read_processed, write_processed

    write_processed(fixture_config)
    frame = read_processed(fixture_config)
    frame.loc[0, "Revenue"] = 2
    report = validate_processed(fixture_config, frame=frame)
    assert not report.ok
    assert read_feature_schema(fixture_config)["features"]


def _config():
    from conftest import BASE_CONFIG

    from shopper_intent.config import load_config

    return load_config(BASE_CONFIG)
