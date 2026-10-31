"""Offline fixtures: a small synthetic session table and a config pointing at it.

No test touches the network. The fixture is generated in code with a fixed seed,
so the same rows and the same metrics come back on every machine.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = REPO_ROOT / "configs" / "base.yaml"

MONTHS = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def make_sessions(rows: int = 120, seed: int = 7) -> pd.DataFrame:
    """Synthetic session table with a real signal in PageValues and duration."""
    rng = np.random.default_rng(seed)
    product_related = rng.integers(1, 45, size=rows)
    product_duration = product_related * rng.uniform(5.0, 90.0, size=rows)
    page_values = rng.gamma(shape=1.1, scale=9.0, size=rows) * (rng.random(rows) < 0.55)
    logging = -2.4 + 0.055 * page_values + 0.004 * product_duration / 10.0
    probability = 1.0 / (1.0 + np.exp(-logging))
    revenue = rng.random(rows) < probability
    month = rng.choice(MONTHS, size=rows, p=[0.08, 0.14, 0.2, 0.08, 0.06, 0.12, 0.07, 0.13, 0.07, 0.05])
    visitor = rng.choice(
        ["Returning_Visitor", "New_Visitor", "Other"], size=rows, p=[0.85, 0.14, 0.01]
    )
    return pd.DataFrame(
        {
            "Administrative": rng.integers(0, 12, size=rows),
            "Administrative_Duration": rng.gamma(2.0, 30.0, size=rows).round(3),
            "Informational": rng.integers(0, 6, size=rows),
            "Informational_Duration": rng.gamma(1.5, 25.0, size=rows).round(3),
            "ProductRelated": product_related,
            "ProductRelated_Duration": product_duration.round(3),
            "BounceRates": rng.uniform(0.0, 0.2, size=rows).round(4),
            "ExitRates": rng.uniform(0.0, 0.2, size=rows).round(4),
            "PageValues": page_values.round(4),
            "SpecialDay": rng.choice([0.0, 0.2, 0.6, 1.0], size=rows, p=[0.9, 0.04, 0.03, 0.03]),
            "Month": month,
            "OperatingSystems": rng.integers(1, 9, size=rows),
            "Browser": rng.integers(1, 14, size=rows),
            "Region": rng.integers(1, 10, size=rows),
            "TrafficType": rng.integers(1, 21, size=rows),
            "VisitorType": visitor,
            "Weekend": rng.random(rows) < 0.23,
            "Revenue": revenue,
        }
    )


def write_manifest(raw_path: Path, rows: int) -> dict:
    frame = pd.read_csv(raw_path)
    digest = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    manifest = {
        "dataset": "synthetic fixture",
        "source_url": "fixture://synthetic",
        "primary_source_url": "fixture://synthetic",
        "fallback_used": False,
        "fetched_at": "2026-01-01T00:00:00+00:00",
        "sha256": digest,
        "bytes": raw_path.stat().st_size,
        "row_count": rows,
        "column_count": len(frame.columns),
        "columns": list(frame.columns),
        "first_observation": json.loads(frame.iloc[0].to_json()),
        "last_observation": json.loads(frame.iloc[-1].to_json()),
        "license": "synthetic",
    }
    (raw_path.parent / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def write_config(root: Path, rows: int = 120, model_overrides: dict | None = None) -> Path:
    config = yaml.safe_load(BASE_CONFIG.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    config["data"]["min_rows"] = 50
    config["features"]["test_size"] = 0.25
    config["model"]["cv_folds"] = 5
    config["model"]["lightgbm"].update(
        {"n_estimators": 60, "num_leaves": 7, "min_child_samples": 10, "learning_rate": 0.1}
    )
    if model_overrides:
        config["model"]["lightgbm"].update(model_overrides)
    config["business"]["threshold_candidates"] = 100

    directory = root / "configs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "base.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    frame = make_sessions(rows=rows)
    raw_path = raw_dir / config["data"]["raw_filename"]
    frame.to_csv(raw_path, index=False)
    write_manifest(raw_path, len(frame))
    return path


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("shopper_fixture")


@pytest.fixture(scope="session")
def fixture_config_path(fixture_root: Path) -> Path:
    return write_config(fixture_root)


@pytest.fixture(scope="session")
def fixture_config(fixture_config_path: Path):
    from shopper_intent.config import load_config

    return load_config(fixture_config_path)


@pytest.fixture(scope="session")
def raw_sessions() -> pd.DataFrame:
    return make_sessions()


@pytest.fixture(scope="session")
def trained(fixture_config):
    """Processed data plus one trained artifact per model, built once per session."""
    from shopper_intent.features.build import write_processed
    from shopper_intent.models.train import train

    write_processed(fixture_config)
    results = train(fixture_config, version="test-0001")
    return {"config": fixture_config, "results": results}


@pytest.fixture()
def api_client(trained):
    from fastapi.testclient import TestClient

    from shopper_intent.serving.app import create_app

    with TestClient(create_app(trained["config"])) as client:
        yield client
