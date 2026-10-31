"""Model factory: one name in, one unfitted estimator out.

Optional dependencies are resolved at call time with ``importlib.util.find_spec``
so the core install (`logistic`, `lightgbm`) keeps working when XGBoost or
CatBoost are absent. Asking for a model whose package is missing raises
``MissingDependency`` naming the extra to install, rather than an ImportError
from deep inside a helper.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shopper_intent.config import MODEL_NAMES, Settings

#: Import module to distribution name. ``find_spec`` needs the module name,
#: and scikit-learn imports as ``sklearn``, so the two cannot be the same map.
REQUIRED_MODULES = {
    "logistic": "sklearn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
}

DISTRIBUTIONS = {
    "logistic": "scikit-learn",
    "lightgbm": "lightgbm",
    "xgboost": "xgboost",
    "catboost": "catboost",
}

#: Models fitted on the native table (raw categorical columns, no one-hot).
NATIVE_CATEGORICAL_TYPES = ("xgboost", "catboost")


class MissingDependency(RuntimeError):
    """Raised when a model is requested but its optional package is not installed."""


def is_available(model_type: str) -> bool:
    module = REQUIRED_MODULES.get(model_type)
    if module is None:
        return False
    return importlib.util.find_spec(module) is not None


def available_models() -> list[str]:
    return [name for name in MODEL_NAMES if is_available(name)]


def require(model_type: str) -> None:
    if model_type not in MODEL_NAMES:
        raise ValueError(f"unknown model type {model_type!r}; expected one of {MODEL_NAMES}")
    if not is_available(model_type):
        package = DISTRIBUTIONS[model_type]
        raise MissingDependency(
            f"model {model_type!r} needs {package}; install it with the 'modern' extra"
        )


def uses_native_categorical(model_type: str) -> bool:
    return model_type in NATIVE_CATEGORICAL_TYPES


def categorical_format(model_type: str) -> str:
    """XGBoost wants pandas ``category`` dtype; CatBoost wants plain strings."""
    return "category" if model_type == "xgboost" else "str"


def positive_weight(target: np.ndarray) -> float:
    """Negative to positive ratio, the usual ``scale_pos_weight`` for imbalance."""
    y = np.asarray(target).astype(int)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0:
        return 1.0
    return float(negatives / positives)


def build_estimator(
    config: Settings,
    model_type: str,
    seed: int,
    target: np.ndarray | None = None,
) -> Pipeline:
    """Fresh, unfitted pipeline for one model type.

    ``target`` is only needed by XGBoost when ``scale_pos_weight`` is ``auto``;
    it is the labels of the fold being fitted, so the weight never sees a row
    from the validation fold.
    """
    require(model_type)
    if model_type == "logistic":
        params = config.model.logistic
        classifier = LogisticRegression(
            C=params.C,
            max_iter=params.max_iter,
            class_weight=None if params.class_weight == "none" else params.class_weight,
            random_state=seed,
        )
        return Pipeline([("scaler", StandardScaler()), ("clf", classifier)])

    if model_type == "lightgbm":
        from lightgbm import LGBMClassifier

        params = config.model.lightgbm.model_dump()
        classifier = LGBMClassifier(
            **params,
            random_state=seed,
            n_jobs=1,
            deterministic=True,
            force_col_wise=True,
            verbosity=-1,
        )
        return Pipeline([("clf", classifier)])

    if model_type == "xgboost":
        from xgboost import XGBClassifier

        params = config.model.xgboost
        weight = params.scale_pos_weight
        if isinstance(weight, str) and weight == "auto":
            weight = positive_weight(target) if target is not None else 1.0
        classifier = XGBClassifier(
            n_estimators=params.n_estimators,
            max_depth=params.max_depth,
            learning_rate=params.learning_rate,
            subsample=params.subsample,
            colsample_bytree=params.colsample_bytree,
            min_child_weight=params.min_child_weight,
            reg_lambda=params.reg_lambda,
            tree_method=params.tree_method,
            enable_categorical=params.enable_categorical,
            scale_pos_weight=float(weight),
            objective="binary:logistic",
            eval_metric="aucpr",
            n_jobs=1,
            random_state=seed,
        )
        return Pipeline([("clf", classifier)])

    from catboost import CatBoostClassifier

    params = config.model.catboost
    # CatBoost rejects subsample under Bayesian bootstrap, so the option is only
    # passed when the configured bootstrap actually uses row subsampling.
    sampling = (
        {"subsample": params.subsample} if params.bootstrap_type == "Bernoulli" else {}
    )
    classifier = CatBoostClassifier(
        iterations=params.iterations,
        depth=params.depth,
        learning_rate=params.learning_rate,
        l2_leaf_reg=params.l2_leaf_reg,
        auto_class_weights=None if params.auto_class_weights == "None" else params.auto_class_weights,
        eval_metric=params.eval_metric,
        bootstrap_type=params.bootstrap_type,
        boosting_type=params.boosting_type,
        one_hot_max_size=params.one_hot_max_size,
        random_strength=params.random_strength,
        thread_count=params.thread_count,
        verbose=params.verbose,
        allow_writing_files=False,
        cat_features=list(config.features.categorical),
        random_seed=seed,
        **sampling,
    )
    return Pipeline([("clf", classifier)])
