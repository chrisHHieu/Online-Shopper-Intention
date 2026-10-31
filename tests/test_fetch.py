"""Fetch cache behaviour and manifest contents, without any network access."""

from __future__ import annotations

import json

import pytest
from conftest import make_sessions, write_config

from shopper_intent.config import load_config
from shopper_intent.data import fetch as fetch_module
from shopper_intent.data.fetch import FetchError, cached_manifest, fetch


def block_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(fetch_module, "_download", fail)


def test_manifest_records_provenance(fixture_config) -> None:
    manifest = cached_manifest(fixture_config)
    assert manifest is not None
    assert manifest["row_count"] == 120
    assert manifest["column_count"] == 18
    assert len(manifest["sha256"]) == 64
    assert manifest["source_url"] == "fixture://synthetic"
    assert "PageValues" in manifest["first_observation"]


def test_cached_fetch_makes_no_request(fixture_config, monkeypatch) -> None:
    block_network(monkeypatch)
    manifest = fetch(fixture_config)
    assert manifest["row_count"] == 120
    assert manifest["sha256"] == cached_manifest(fixture_config)["sha256"]


def test_changed_file_forces_a_request(tmp_path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    with config.raw_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    assert cached_manifest(config) is None
    block_network(monkeypatch)
    with pytest.raises(AssertionError, match="network access attempted"):
        fetch(config)


def test_download_failure_is_reported(tmp_path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    config.raw_path.unlink()

    def fail(*args, **kwargs):
        raise FetchError("upstream unavailable")

    monkeypatch.setattr(fetch_module, "_download", fail)
    with pytest.raises(FetchError, match="upstream unavailable"):
        fetch(config)


def test_force_redownloads_and_rewrites_manifest(tmp_path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    payload = make_sessions(rows=120).to_csv(index=False).encode("utf-8")
    monkeypatch.setattr(fetch_module, "_download", lambda *args, **kwargs: payload)

    manifest = fetch(config, force=True)
    assert manifest["row_count"] == 120
    assert manifest["source_url"] == config.data.source_url
    assert manifest["fallback_used"] is False
    saved = json.loads((config.raw_path.parent / "manifest.json").read_text(encoding="utf-8"))
    assert saved["sha256"] == manifest["sha256"]
    assert config.raw_path.read_bytes() == payload


def test_payload_without_expected_columns_is_rejected(tmp_path, monkeypatch) -> None:
    config = load_config(write_config(tmp_path))
    config.raw_path.unlink()
    monkeypatch.setattr(
        fetch_module,
        "_download",
        lambda *args, **kwargs: b"a,b\n1,2\n",
    )
    with pytest.raises(FetchError, match="missing columns"):
        fetch(config, force=True)
