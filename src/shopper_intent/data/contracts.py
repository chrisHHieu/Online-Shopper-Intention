"""Pandera data contracts for the raw table and the processed matrix.

Pandera is an optional dependency: without it the declarative contract in
``data.validate`` still runs, and this module reports that it was skipped rather
than failing the import. Validation is lazy so one call reports every broken
column instead of only the first.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd

from shopper_intent.config import Settings
from shopper_intent.data.validate import DEFAULT_SCHEMA

PANDERA_AVAILABLE = importlib.util.find_spec("pandera") is not None

RAW_CATEGORICAL = ("Month", "VisitorType", "Weekend")


def _pa() -> Any:
    if not PANDERA_AVAILABLE:
        raise RuntimeError("pandera is not installed; install the 'modern' extra")
    import pandera.pandas as pa

    return pa


def _column(pa: Any, name: str, rule: dict[str, Any]) -> Any:
    dtype = rule.get("dtype", "float")
    checks = []
    if dtype == "int":
        base = pa.Int64
        checks.append(pa.Check.ge(rule["min"]))
        checks.append(pa.Check.le(rule["max"]))
    elif dtype == "float":
        base = pa.Float64
        if "min" in rule:
            checks.append(pa.Check.ge(rule["min"]))
        if "max" in rule:
            checks.append(pa.Check.le(rule["max"]))
    elif dtype == "bool":
        base = pa.Bool
    else:
        base = pa.String
        checks.append(pa.Check.isin(rule.get("allowed", [])))
    return pa.Column(base, checks=checks, nullable=False, name=name)


def raw_schema(schema: dict[str, dict[str, Any]] | None = None) -> Any:
    """Strict contract for the raw CSV: every published column, no extras."""
    pa = _pa()
    rules = DEFAULT_SCHEMA if schema is None else schema
    return pa.DataFrameSchema(
        {name: _column(pa, name, rule) for name, rule in rules.items()},
        strict=True,
        coerce=True,
    )


def processed_schema(config: Settings, features: list[str]) -> Any:
    """Contract for the processed table that training and serving rely on."""
    pa = _pa()
    columns = {
        name: pa.Column(pa.Float64, checks=[pa.Check.ge(0.0)], nullable=False)
        for name in features
    }
    if "BounceExitSpread" in columns:
        # Exit rate minus bounce rate is the only signed feature.
        columns["BounceExitSpread"] = pa.Column(pa.Float64, nullable=False)
    columns["Revenue"] = pa.Column(
        pa.Int64, checks=[pa.Check.isin([0, 1])], nullable=False
    )
    columns["split"] = pa.Column(
        pa.String, checks=[pa.Check.isin(["train", "test"])], nullable=False
    )
    return pa.DataFrameSchema(columns, strict=True, coerce=True)


def _failure_rows(error: Exception) -> list[dict[str, Any]]:
    cases = getattr(error, "failure_cases", None)
    if cases is None or not isinstance(cases, pd.DataFrame):
        return []
    rows = cases.to_dict(orient="records")
    return rows


def check_raw(frame: pd.DataFrame, schema: dict[str, dict[str, Any]] | None = None):
    """Run the pandera contract, returning ``(name, status, detail)`` triples."""
    from shopper_intent.data.validate import Check

    if not PANDERA_AVAILABLE:
        return [Check("pandera:raw", "skip", "pandera is not installed")]
    try:
        raw_schema(schema).validate(frame, lazy=True)
    except Exception as error:  # noqa: BLE001 - SchemaErrors carries the detail
        rows = _failure_rows(error)
        if not rows:
            return [Check("pandera:raw", "fail", str(error).splitlines()[0][:200])]
        findings = []
        for row in rows[:50]:
            name = f"pandera:{row.get('column', 'frame')}:{row.get('check', 'check')}"
            failure = str(row.get("failure_case", ""))[:60]
            findings.append(Check(name, "fail", f"failure case {failure}"))
        if len(rows) > 50:
            findings.append(Check("pandera:raw", "warn", f"{len(rows) - 50} more failures"))
        return findings
    return [Check("pandera:raw", "pass", f"{frame.shape[1]} columns, {len(frame)} rows")]


def check_processed(frame: pd.DataFrame, config: Settings, features: list[str]):
    """Run the processed contract on the table written by ``features``."""
    from shopper_intent.data.validate import Check

    if not PANDERA_AVAILABLE:
        return [Check("pandera:processed", "skip", "pandera is not installed")]
    try:
        processed_schema(config, features).validate(frame, lazy=True)
    except Exception as error:  # noqa: BLE001 - SchemaErrors carries the detail
        rows = _failure_rows(error)
        if not rows:
            return [Check("pandera:processed", "fail", str(error).splitlines()[0][:200])]
        return [
            Check(
                f"pandera:{row.get('column', 'frame')}:{row.get('check', 'check')}",
                "fail",
                f"failure case {str(row.get('failure_case', ''))[:60]}",
            )
            for row in rows[:50]
        ]
    return [
        Check(
            "pandera:processed",
            "pass",
            f"{len(features)} feature columns, {len(frame)} rows",
        )
    ]
