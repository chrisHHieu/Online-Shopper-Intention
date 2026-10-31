"""Per-row contributions and readable reason codes.

Tree models are explained with ``shap.TreeExplainer``. When SHAP is not
installed the native booster contribution APIs are used instead, so the core
install still produces explanations. Logistic regression uses the exact linear
decomposition ``standardised value times coefficient``.

Contributions are additive on the model's own scale (log odds for the tree
models). They explain what the model did with a row, not what would happen if a
value changed, which is a causal question this design does not answer.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
import pandas as pd

#: Plain language for the numeric columns. Categorical dummies are expanded from
#: their prefix, so only the bare column names need an entry.
FEATURE_LABELS = {
    "Administrative": "administrative pages",
    "Administrative_Duration": "time on administrative pages",
    "Informational": "informational pages",
    "Informational_Duration": "time on informational pages",
    "ProductRelated": "product pages viewed",
    "ProductRelated_Duration": "time on product pages",
    "BounceRates": "bounce rate",
    "ExitRates": "exit rate",
    "PageValues": "page value",
    "SpecialDay": "proximity to a special day",
    "OperatingSystems": "operating system code",
    "Browser": "browser code",
    "Region": "region code",
    "TrafficType": "traffic type code",
    "PageValuePerProductPage": "page value per product page",
    "DurationPerPage": "time per page viewed",
    "BounceExitSpread": "exit minus bounce rate",
}

CATEGORICAL_LABELS = {
    "Month": "month",
    "VisitorType": "visitor type",
    "Weekend": "weekend",
}


def shap_available() -> bool:
    return importlib.util.find_spec("shap") is not None


def feature_label(feature: str, vocabulary: dict[str, list[Any]] | None = None) -> str:
    """Human name for a model column, expanding ``Month_May`` to ``month is May``."""
    if feature in FEATURE_LABELS:
        return FEATURE_LABELS[feature]
    for name, categories in (vocabulary or CATEGORICAL_LABELS).items():
        prefix = f"{name}_"
        if feature.startswith(prefix):
            value = feature[len(prefix):]
            return f"{CATEGORICAL_LABELS.get(name, name.lower())} is {value}"
    for name in CATEGORICAL_LABELS:
        prefix = f"{name}_"
        if feature.startswith(prefix):
            return f"{CATEGORICAL_LABELS[name]} is {feature[len(prefix):]}"
    return feature.replace("_", " ").lower()


def _first_row(values: Any) -> np.ndarray:
    """Normalise the shapes SHAP returns across versions and model families."""
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[:, :, 1]
    return array


def _tree_shap(final: Any, matrix: pd.DataFrame) -> np.ndarray:
    import shap

    explainer = shap.TreeExplainer(final)
    values = explainer.shap_values(matrix)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    return _first_row(values)


def _booster_fallback(model_type: str, final: Any, matrix: pd.DataFrame) -> np.ndarray:
    if model_type == "lightgbm":
        return np.asarray(final.booster_.predict(matrix, pred_contrib=True), dtype=float)
    if model_type == "xgboost":
        import xgboost as xgb

        booster = final.get_booster()
        contribution = booster.predict(
            xgb.DMatrix(matrix, enable_categorical=True), pred_contribs=True
        )
        return np.asarray(contribution, dtype=float)
    from catboost import Pool

    pool = Pool(matrix, cat_features=list(matrix.select_dtypes(include="object").columns))
    return np.asarray(final.get_feature_importance(pool, type="ShapValues"), dtype=float)


def contribution_matrix(model: Any, matrix: pd.DataFrame) -> tuple[np.ndarray, str]:
    """Additive contribution per column for every row, plus the engine used."""
    final = model.estimator.steps[-1][1]
    columns = len(model.feature_names)

    if hasattr(final, "coef_"):
        transformed = np.asarray(model._preprocess(matrix), dtype=float)
        return transformed * np.asarray(final.coef_[0], dtype=float), "linear"

    worked = matrix
    if model.model_type == "xgboost":
        worked = matrix.apply(lambda column: column.cat.codes if str(column.dtype) == "category" else column)
    if shap_available():
        try:
            values = _tree_shap(final, worked)
            return np.asarray(values, dtype=float)[:, :columns], "shap.TreeExplainer"
        except Exception:  # noqa: BLE001 - fall back to the booster, never fail a prediction
            pass
    values = _booster_fallback(model.model_type, final, worked)
    return np.asarray(values, dtype=float)[:, :columns], f"{model.model_type}.native"


def top_reasons(
    model: Any,
    matrix: pd.DataFrame,
    top_k: int = 5,
    vocabulary: dict[str, list[Any]] | None = None,
) -> list[list[dict[str, Any]]]:
    """Top-k contributions per row with a sentence a person can read."""
    contributions, engine = contribution_matrix(model, matrix)
    values = matrix.to_numpy(dtype=object)
    explanations: list[list[dict[str, Any]]] = []
    for row_index, row in enumerate(contributions):
        order = np.argsort(-np.abs(row))[:top_k]
        entries = []
        for index in order:
            feature = model.feature_names[index]
            contribution = float(row[index])
            raw_value = values[row_index, index]
            try:
                numeric_value: float | None = float(raw_value)
            except (TypeError, ValueError):
                numeric_value = None
            label = feature_label(feature, vocabulary)
            entries.append(
                {
                    "feature": feature,
                    "contribution": round(contribution, 5),
                    "direction": "increases" if contribution >= 0 else "decreases",
                    "value": round(numeric_value, 4) if numeric_value is not None else None,
                    "value_label": None if numeric_value is not None else str(raw_value),
                    "reason": (
                        f"{label} increases the estimate"
                        if contribution >= 0
                        else f"{label} decreases the estimate"
                    ),
                    "engine": engine,
                }
            )
        explanations.append(entries)
    return explanations
