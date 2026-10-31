"""Configuration loading, hashing and path resolution."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from conftest import BASE_CONFIG, REPO_ROOT
from pydantic import ValidationError

from shopper_intent.config import load_config, project_root


def test_loads_repository_config() -> None:
    config = load_config(BASE_CONFIG)
    assert config.project.name == "online-shopper-intention"
    assert config.model.seed == 42
    assert config.model.cv_folds == 5
    assert config.business.value_per_conversion == 100.0
    assert config.business.cost_per_outreach == 1.0
    assert config.root == REPO_ROOT
    assert len(config.config_hash()) == 64


def test_relative_paths_resolve_against_root() -> None:
    config = load_config(BASE_CONFIG)
    assert config.raw_path == REPO_ROOT / "data" / "raw" / config.data.raw_filename
    assert config.processed_path == REPO_ROOT / "data" / "processed" / "sessions.csv"
    assert config.models_dir == REPO_ROOT / "models"


def test_explicit_root_override(tmp_path: Path) -> None:
    config = load_config(BASE_CONFIG, root=tmp_path)
    assert config.root == tmp_path
    assert config.raw_path == tmp_path / "data" / "raw" / "online_shoppers_intention.csv"


def test_config_hash_is_stable_and_sensitive(tmp_path: Path) -> None:
    first = load_config(BASE_CONFIG)
    second = load_config(BASE_CONFIG)
    assert first.config_hash() == second.config_hash()

    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["business"]["value_per_conversion"] = 250.0
    changed_path = tmp_path / "base.yaml"
    changed_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    changed = load_config(changed_path, root=REPO_ROOT)
    assert changed.config_hash() != first.config_hash()


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["model"]["unsupported_option"] = 3
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_invalid_threshold_order_is_rejected(tmp_path: Path) -> None:
    from shopper_intent.config import load_config as reload_config

    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["calibration"]["clip_low"] = 0.9
    payload["calibration"]["clip_high"] = 0.8
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        reload_config(path)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_categorical_vocabulary_keeps_booleans() -> None:
    config = load_config(BASE_CONFIG)
    assert config.features.categorical["Weekend"] == [False, True]
    assert "Jan" not in config.features.categorical["Month"]


def test_project_root_contains_pyproject() -> None:
    assert (project_root() / "pyproject.toml").is_file()


def test_candidate_defaults_stay_light() -> None:
    config = load_config(BASE_CONFIG)
    assert config.model.candidates == ["logistic", "lightgbm"]
    assert config.serving.model in config.model.candidates


def test_full_config_has_four_candidates() -> None:
    config = load_config(REPO_ROOT / "configs" / "full.yaml")
    assert config.model.candidates == ["logistic", "lightgbm", "xgboost", "catboost"]


def test_modern_stack_defaults() -> None:
    config = load_config(BASE_CONFIG)
    assert config.calibration.method == "auto"
    assert config.conformal.enabled is True
    assert 0.0 < config.conformal.alpha < 1.0
    assert config.tracking.enabled is False
    assert config.tracking.tracking_uri.startswith("file:")
    assert config.tuning.trials >= config.tuning.fast_trials
    assert config.optuna_storage == REPO_ROOT / "reports" / "optuna.db"
    assert config.mlruns_dir == REPO_ROOT / "mlruns"


def test_unknown_candidate_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["model"]["candidates"] = ["logistic", "tabpfn"]
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_serving_model_must_be_a_candidate(tmp_path: Path) -> None:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["serving"]["model"] = "catboost"
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError, match="not in model.candidates"):
        load_config(path)


def test_unknown_calibration_method_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["calibration"]["method"] = "beta"
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(path)


def test_repeated_candidate_is_rejected(tmp_path: Path) -> None:
    payload = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    payload = copy.deepcopy(payload)
    payload["model"]["candidates"] = ["lightgbm", "lightgbm"]
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValidationError, match="must not repeat"):
        load_config(path)
