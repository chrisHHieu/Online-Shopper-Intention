"""Deterministic feature build.

Input: the 17 raw session columns. Output: raw numerics, three ratio features
and fixed-vocabulary one-hot columns for Month, VisitorType and Weekend. Nothing
is computed from ``Revenue``, and the encoder vocabulary comes from config rather
than from the data, so the transform is stable across train and serve.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from shopper_intent.config import Settings
from shopper_intent.data.fetch import load_raw

RATIO_COLUMNS = ["PageValuePerProductPage", "DurationPerPage", "BounceExitSpread"]

RAW_COLUMNS = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]


def _one_hot(frame: pd.DataFrame, column: str, categories: list[Any]) -> pd.DataFrame:
    series = frame[column]
    if all(isinstance(value, bool) for value in categories):
        series = series.astype(bool)
    else:
        series = series.astype(str)
    categorical = pd.Categorical(series, categories=categories)
    encoded = pd.get_dummies(categorical, prefix=column, dtype=np.int8)
    encoded.columns = [f"{column}_{value}" for value in categories]
    return encoded


def build_features(raw: pd.DataFrame, config: Settings) -> pd.DataFrame:
    """Build the model matrix from raw session columns.

    The function is pure with respect to its input: the same frame always
    produces the same output, including column order.
    """
    missing = [column for column in RAW_COLUMNS if column not in raw.columns]
    if missing:
        raise ValueError(f"raw frame is missing columns: {missing}")

    frame = raw[RAW_COLUMNS].copy()
    for column in config.features.numeric:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    for column in config.features.categorical:
        if column not in frame.columns:
            raise ValueError(f"categorical column {column!r} is not in the raw frame")

    pages = frame["Administrative"] + frame["Informational"] + frame["ProductRelated"]
    duration = (
        frame["Administrative_Duration"]
        + frame["Informational_Duration"]
        + frame["ProductRelated_Duration"]
    )
    features = frame[config.features.numeric].copy()
    features["PageValuePerProductPage"] = frame["PageValues"] / (frame["ProductRelated"] + 1.0)
    features["DurationPerPage"] = duration / (pages + 1.0)
    features["BounceExitSpread"] = frame["ExitRates"] - frame["BounceRates"]

    blocks = [features]
    for column, categories in config.features.categorical.items():
        blocks.append(_one_hot(frame, column, list(categories)))
    matrix = pd.concat(blocks, axis=1)
    matrix = matrix.astype("float64")
    if matrix.isna().any().any():
        raise ValueError("feature matrix contains nulls")
    if not np.isfinite(matrix.to_numpy()).all():
        raise ValueError("feature matrix contains infinite values")
    return matrix


def feature_columns(config: Settings) -> list[str]:
    """Column order of the model matrix, computed without touching the data."""
    placeholder = pd.DataFrame({column: [0] for column in RAW_COLUMNS})
    for column, categories in config.features.categorical.items():
        if all(isinstance(value, bool) for value in categories):
            placeholder[column] = False
        else:
            placeholder[column] = str(categories[0])
    return list(build_features(placeholder, config).columns)


def assign_split(frame: pd.DataFrame, config: Settings) -> pd.DataFrame:
    """Stratified train/test assignment, deterministic for a given seed."""
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=config.features.test_size,
        random_state=config.features.seed,
    )
    labels = np.full(len(frame), "train", dtype=object)
    indices = np.arange(len(frame))
    _, test_positions = next(splitter.split(indices, frame[config.features.target].to_numpy()))
    labels[test_positions] = "test"
    result = frame.copy()
    result["split"] = labels
    return result


def build_training_frame(raw: pd.DataFrame, config: Settings) -> pd.DataFrame:
    """Feature matrix plus target and split label."""
    if config.features.target not in raw.columns:
        raise ValueError(f"raw frame has no target column {config.features.target!r}")
    matrix = build_features(raw, config)
    target = raw[config.features.target].astype(bool).astype("int8")
    matrix[config.features.target] = target.to_numpy()
    return assign_split(matrix, config)


def schema_path(config: Settings) -> Path:
    return config.processed_dir / "feature_schema.json"


def write_processed(config: Settings, raw: pd.DataFrame | None = None) -> Path:
    raw = load_raw(config) if raw is None else raw
    frame = build_training_frame(raw, config)
    config.processed_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(config.processed_path, index=False)
    schema = {
        "created_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "target": config.features.target,
        "features": [column for column in frame.columns if column not in {"Revenue", "split"}],
        "ratio_features": RATIO_COLUMNS,
        "categorical_vocabulary": config.features.categorical,
        "rows": int(len(frame)),
        "train_rows": int((frame["split"] == "train").sum()),
        "test_rows": int((frame["split"] == "test").sum()),
        "positive_rate": float(frame[config.features.target].mean()),
        "seed": config.features.seed,
    }
    schema_path(config).write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    return config.processed_path


def read_processed(config: Settings) -> pd.DataFrame:
    if not config.processed_path.is_file():
        raise FileNotFoundError(
            f"processed data not found at {config.processed_path}; run 'features' first"
        )
    return pd.read_csv(config.processed_path)


def read_feature_schema(config: Settings) -> dict[str, Any]:
    path = schema_path(config)
    if not path.is_file():
        raise FileNotFoundError(f"feature schema not found at {path}; run 'features' first")
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_for_model(raw: pd.DataFrame, config: Settings) -> pd.DataFrame:
    """Align serving input with the exact column order used in training."""
    matrix = build_features(raw, config)
    expected = read_feature_schema(config)["features"]
    if list(matrix.columns) != expected:
        matrix = matrix.reindex(columns=expected, fill_value=0.0)
    return matrix
