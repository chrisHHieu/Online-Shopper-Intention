"""Data contract for the raw session table.

Every rule is declarative so it can be monkey-patched in tests: a column rule
carries a dtype, an optional inclusive range and an optional set of allowed
values. Failures are errors, softer observations are warnings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from shopper_intent.config import Settings
from shopper_intent.data.fetch import load_raw

DEFAULT_SCHEMA: dict[str, dict[str, Any]] = {
    "Administrative": {"dtype": "int", "min": 0, "max": 1000},
    "Administrative_Duration": {"dtype": "float", "min": 0.0, "max": 1_000_000.0},
    "Informational": {"dtype": "int", "min": 0, "max": 1000},
    "Informational_Duration": {"dtype": "float", "min": 0.0, "max": 1_000_000.0},
    "ProductRelated": {"dtype": "int", "min": 0, "max": 10_000},
    "ProductRelated_Duration": {"dtype": "float", "min": 0.0, "max": 10_000_000.0},
    "BounceRates": {"dtype": "float", "min": 0.0, "max": 1.0},
    "ExitRates": {"dtype": "float", "min": 0.0, "max": 1.0},
    "PageValues": {"dtype": "float", "min": 0.0, "max": 10_000.0},
    "SpecialDay": {"dtype": "float", "min": 0.0, "max": 1.0},
    "Month": {"dtype": "category", "allowed": ["Feb", "Mar", "May", "June", "Jul", "Aug",
                                               "Sep", "Oct", "Nov", "Dec"]},
    "OperatingSystems": {"dtype": "int", "min": 1, "max": 20},
    "Browser": {"dtype": "int", "min": 1, "max": 20},
    "Region": {"dtype": "int", "min": 1, "max": 20},
    "TrafficType": {"dtype": "int", "min": 1, "max": 30},
    "VisitorType": {"dtype": "category", "allowed": ["Returning_Visitor", "New_Visitor", "Other"]},
    "Weekend": {"dtype": "bool", "allowed": [True, False]},
    "Revenue": {"dtype": "bool", "allowed": [True, False]},
}


@dataclass
class Check:
    name: str
    status: str
    detail: str


@dataclass
class ValidationReport:
    checks: list[Check] = field(default_factory=list)
    row_count: int = 0
    column_count: int = 0
    positive_rate: float | None = None

    @property
    def ok(self) -> bool:
        return all(check.status != "fail" for check in self.checks)

    @property
    def failures(self) -> list[Check]:
        return [check for check in self.checks if check.status == "fail"]

    @property
    def warnings(self) -> list[Check]:
        return [check for check in self.checks if check.status == "warn"]

    def add(self, name: str, status: str, detail: str) -> None:
        self.checks.append(Check(name=name, status=status, detail=detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "positive_rate": self.positive_rate,
            "failures": len(self.failures),
            "warnings": len(self.warnings),
            "checks": [
                {"name": c.name, "status": c.status, "detail": c.detail} for c in self.checks
            ],
        }

    def render(self) -> str:
        lines = [f"rows={self.row_count} columns={self.column_count}"]
        if self.positive_rate is not None:
            lines.append(f"positive rate={self.positive_rate:.4f}")
        for check in self.checks:
            lines.append(f"{check.status.upper():4s}  {check.name:<22s}  {check.detail}")
        lines.append("validation passed" if self.ok else "validation failed")
        return "\n".join(lines)


def _is_bool_like(series: pd.Series, allowed: list[Any]) -> bool:
    values = set(series.dropna().unique().tolist())
    normalised = {bool(value) if str(value).lower() in {"true", "false"} else value for value in values}
    return normalised.issubset(set(allowed))


def validate_frame(
    frame: pd.DataFrame,
    config: Settings,
    schema: dict[str, dict[str, Any]] | None = None,
) -> ValidationReport:
    """Apply the data contract and collect every finding, not just the first."""
    schema = DEFAULT_SCHEMA if schema is None else schema
    report = ValidationReport(row_count=len(frame), column_count=frame.shape[1])

    missing = [column for column in schema if column not in frame.columns]
    if missing:
        report.add("schema", "fail", f"missing columns: {missing}")
    else:
        report.add("schema", "pass", f"all {len(schema)} required columns present")

    extra = [column for column in frame.columns if column not in schema]
    if extra:
        report.add("extra_columns", "warn", f"unexpected columns: {extra}")

    if "Revenue" in frame.columns:
        revenue = frame["Revenue"]
        if not _is_bool_like(revenue, [True, False]):
            report.add("target_type", "fail", "Revenue is not boolean-like")
        else:
            rate = float(revenue.astype(bool).mean())
            report.positive_rate = rate
            low, high = config.data.min_positive_rate, config.data.max_positive_rate
            if low <= rate <= high:
                report.add("target_rate", "pass", f"positive rate {rate:.4f} within [{low}, {high}]")
            else:
                report.add("target_rate", "fail", f"positive rate {rate:.4f} outside [{low}, {high}]")

    for column, rule in schema.items():
        if column not in frame.columns:
            continue
        series = frame[column]
        dtype = rule.get("dtype", "float")

        nulls = int(series.isna().sum())
        if nulls:
            report.add(f"nulls:{column}", "fail", f"{nulls} null values")

        if dtype in {"int", "float"}:
            numeric = pd.to_numeric(series, errors="coerce")
            bad = int(numeric.isna().sum() - nulls)
            if bad > 0:
                report.add(f"dtype:{column}", "fail", f"{bad} non-numeric values")
                continue
            if dtype == "int" and not numeric.dropna().mod(1).eq(0).all():
                report.add(f"dtype:{column}", "fail", "fractional values in an integer column")
                continue
            lowest = rule.get("min")
            highest = rule.get("max")
            if lowest is not None and numeric.min() < lowest:
                report.add(f"range:{column}", "fail", f"min {numeric.min()} below {lowest}")
            if highest is not None and numeric.max() > highest:
                report.add(f"range:{column}", "fail", f"max {numeric.max()} above {highest}")
        elif dtype == "category":
            allowed = set(rule.get("allowed", []))
            seen = set(series.dropna().astype(str).unique())
            unexpected = sorted(seen - allowed)
            if unexpected:
                report.add(f"values:{column}", "fail", f"unexpected values: {unexpected}")
        elif dtype == "bool":
            if not _is_bool_like(series, rule.get("allowed", [True, False])):
                report.add(f"values:{column}", "fail", "values are not boolean-like")

    numeric_columns = [c for c, r in schema.items() if r.get("dtype") in {"int", "float"}]
    present_numeric = [c for c in numeric_columns if c in frame.columns]
    if present_numeric:
        infinite = int(frame[present_numeric].isin([float("inf"), float("-inf")]).sum().sum())
        if infinite:
            report.add("finite", "fail", f"{infinite} infinite values")

    duplicates = int(frame.duplicated().sum())
    fraction = duplicates / len(frame) if len(frame) else 0.0
    duplicate_status = "warn" if fraction > config.data.max_duplicate_fraction else "pass"
    report.add(
        "duplicates",
        duplicate_status,
        f"{duplicates} duplicate rows ({fraction:.2%}), limit {config.data.max_duplicate_fraction:.2%}",
    )

    if report.row_count < config.data.min_rows:
        report.add("row_count", "fail", f"{report.row_count} rows, need {config.data.min_rows}")
    else:
        report.add("row_count", "pass", f"{report.row_count} rows, need {config.data.min_rows}")

    return report


def validate(config: Settings, schema: dict[str, dict[str, Any]] | None = None) -> ValidationReport:
    return validate_frame(load_raw(config), config, schema=schema)
