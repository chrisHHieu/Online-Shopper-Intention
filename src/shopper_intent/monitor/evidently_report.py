"""Evidently drift report, with the month proxy the data allows.

The UCI table has no timestamps. ``Month`` is the only time-like column, so the
report splits the sessions into an early period (February to August) and a late
period (September to December) and treats the second as "new traffic". That is a
proxy for a temporal split, not a temporal split: sessions from the same calendar
year are mixed within each period, and the order of months is an assumption from
the published dataset description.

Evidently reports feature drift per column (KS for numeric, Z-test for
categorical). Target drift is not a preset in Evidently 0.7, so the conversion
rate shift is reported next to the feature table together with its PSI.

The HTML report is the artifact a person reads; the JSON next to it is what the
CLI prints and what a scheduler would check.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from shopper_intent.config import Settings
from shopper_intent.data.fetch import load_raw
from shopper_intent.features.build import build_features
from shopper_intent.monitor.drift import psi

EVIDENTLY_AVAILABLE = importlib.util.find_spec("evidently") is not None

MONTH_ORDER = ["Feb", "Mar", "Apr", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DEFAULT_CUT = "Sep"


class EvidentlyUnavailable(RuntimeError):
    """Raised when the report is requested without the evidently package."""


def month_periods(
    raw: pd.DataFrame,
    reference_months: list[str] | None = None,
    current_months: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Months present in the data, split at September unless told otherwise."""
    present = [month for month in MONTH_ORDER if month in set(raw["Month"].astype(str))]
    if reference_months and current_months:
        return list(reference_months), list(current_months)
    cut = present.index(DEFAULT_CUT) if DEFAULT_CUT in present else max(1, len(present) - 1)
    return present[:cut], present[cut:]


def _summarise(result: dict[str, Any]) -> dict[str, Any]:
    """Pull per-column drift out of Evidently's metric list."""
    columns: list[dict[str, Any]] = []
    total: dict[str, Any] = {}
    for metric in result.get("metrics", []):
        config = metric.get("config", {})
        kind = str(config.get("type", ""))
        if kind.endswith("DriftedColumnsCount"):
            total = dict(metric.get("value", {}))
            continue
        if not kind.endswith("ValueDrift"):
            continue
        column = config.get("column")
        value = metric.get("value")
        threshold = config.get("threshold", 0.05)
        if column is None or not isinstance(value, (int, float)):
            continue
        columns.append(
            {
                "column": column,
                "method": config.get("method"),
                "value": float(value),
                "threshold": float(threshold),
                "drifted": bool(value < threshold),
            }
        )
    columns.sort(key=lambda row: (not row["drifted"], row["value"]))
    return {
        "columns_checked": len(columns),
        "drifted_columns": [row["column"] for row in columns if row["drifted"]],
        "drifted_count": int(total.get("count", sum(row["drifted"] for row in columns))),
        "drifted_share": float(total.get("share", 0.0)),
        "details": columns,
    }


def drift_evidently(
    config: Settings,
    current_path: Path | None = None,
    reference_months: list[str] | None = None,
    current_months: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Run Evidently drift and write the HTML report plus a JSON summary."""
    if not EVIDENTLY_AVAILABLE:
        raise EvidentlyUnavailable(
            "evidently is not installed; install the 'modern' extra or use 'drift' for PSI"
        )
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    if current_path is not None:
        current_raw = pd.read_csv(current_path)
        reference_raw = load_raw(config)
        reference_months = list(reference_months or [])
        current_months = list(current_months or [])
        reference_label = f"training file, {len(reference_raw)} rows"
        current_label = f"{current_path.name}, {len(current_raw)} rows"
    else:
        reference_raw = load_raw(config)
        reference_months, current_months = month_periods(
            reference_raw, reference_months, current_months
        )
        current_raw = reference_raw[reference_raw["Month"].astype(str).isin(current_months)]
        reference_raw = reference_raw[reference_raw["Month"].astype(str).isin(reference_months)]
        reference_label = f"months {', '.join(reference_months)}"
        current_label = f"months {', '.join(current_months)}"

    if reference_raw.empty or current_raw.empty:
        raise ValueError("one of the drift periods is empty; check --reference/--current months")

    features = list(build_features(reference_raw, config).columns)
    reference = build_features(reference_raw, config)
    current = build_features(current_raw, config)

    target = config.features.target
    reference_rate = (
        float(reference_raw[target].astype(bool).mean()) if target in reference_raw else None
    )
    current_rate = (
        float(current_raw[target].astype(bool).mean()) if target in current_raw else None
    )

    definition = DataDefinition(numerical_columns=features, categorical_columns=[])
    report = Report([DataDriftPreset()])
    run = report.run(
        reference_data=Dataset.from_pandas(reference, data_definition=definition),
        current_data=Dataset.from_pandas(current, data_definition=definition),
    )

    config.reports_dir.mkdir(parents=True, exist_ok=True)
    html_path = output or (config.reports_dir / "evidently_drift.html")
    run.save_html(str(html_path))
    summary = _summarise(run.dict())

    target_psi = None
    if reference_rate is not None and current_rate is not None:
        target_psi = psi(
            reference_raw[target].astype(float).to_numpy(),
            current_raw[target].astype(float).to_numpy(),
        )

    payload = {
        "generated_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "engine": "evidently",
        "evidently_version": _version(),
        "reference": reference_label,
        "current": current_label,
        "reference_months": reference_months,
        "current_months": current_months,
        "rows": {"reference": int(len(reference)), "current": int(len(current))},
        "features": summary,
        "target": {
            "column": target,
            "reference_rate": reference_rate,
            "current_rate": current_rate,
            "delta": None
            if reference_rate is None or current_rate is None
            else current_rate - reference_rate,
            "psi": target_psi,
            "note": "target drift is computed here, Evidently 0.7 has no target drift preset",
        },
        "html": str(html_path),
    }
    json_path = config.reports_dir / "evidently_drift.json"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["json"] = str(json_path)
    return payload


def _version() -> str:
    try:
        import evidently

        return str(getattr(evidently, "__version__", "unknown"))
    except Exception:  # noqa: BLE001 - a version string must never fail a report
        return "unknown"
