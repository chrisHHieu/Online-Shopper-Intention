"""Optuna search over the four candidate models.

The objective is the pipeline this repository actually ships: stratified k-fold
out-of-fold scores, a calibrator fitted on those scores, a threshold chosen on
them by expected net value, and the resulting net value per 1000 sessions. It is
maximised, so the search never optimises AUC or accuracy directly.

Because the threshold is picked on the same out-of-fold scores that produce the
objective, the tuned value is an in-sample estimate for ranking configurations.
The held-out number in the report comes from a model trained once with the tuned
parameters, so it is not a search result.

Storage is a local SQLite file so studies resume and can be inspected with
``optuna-dashboard``. Each model gets its own study named ``<prefix>-<model>``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

from shopper_intent.config import Settings
from shopper_intent.features.build import to_native
from shopper_intent.models.calibration import fit_calibrator
from shopper_intent.models.candidates import build_estimator, require
from shopper_intent.models.metrics import choose_threshold

SUGGESTERS: dict[str, Callable[[Any], dict[str, Any]]] = {}


def suggest_logistic(trial: Any) -> dict[str, Any]:
    return {"C": trial.suggest_float("C", 1e-2, 1e2, log=True)}


def suggest_lightgbm(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 8, 64),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 120),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def suggest_xgboost(trial: Any) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 900, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
    }


def suggest_catboost(trial: Any) -> dict[str, Any]:
    return {
        "iterations": trial.suggest_int("iterations", 300, 900, step=50),
        "depth": trial.suggest_int("depth", 4, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 20.0, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
    }


SUGGESTERS.update(
    {
        "logistic": suggest_logistic,
        "lightgbm": suggest_lightgbm,
        "xgboost": suggest_xgboost,
        "catboost": suggest_catboost,
    }
)


def model_matrix(config: Settings, model_type: str, matrix: pd.DataFrame) -> pd.DataFrame:
    """The matrix a given model is fitted on: one-hot or native categorical."""
    if model_type in ("xgboost", "catboost"):
        return to_native(
            matrix, config.features.categorical, as_category=model_type == "xgboost"
        )
    return matrix


def with_params(config: Settings, model_type: str, params: dict[str, Any]) -> Settings:
    """Copy of the config with one model section replaced by trial parameters."""
    section = getattr(config.model, model_type).model_copy(update=params)
    trial_model = config.model.model_copy(update={model_type: section})
    return config.model_copy(update={"model": trial_model})


def suggest(trial: Any, model_type: str) -> dict[str, Any]:
    return SUGGESTERS[model_type](trial)


def objective_for(
    config: Settings,
    model_type: str,
    features: pd.DataFrame,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Callable[[Any], float]:
    """Build the Optuna objective for one model type."""
    value = config.business.value_per_conversion
    cost = config.business.cost_per_outreach
    limit = config.business.threshold_candidates

    def objective(trial: Any) -> float:
        trial_config = with_params(config, model_type, suggest(trial, model_type))
        oof = np.zeros(len(target), dtype=float)
        running: list[float] = []
        for step, (train_index, valid_index) in enumerate(folds):
            estimator = build_estimator(
                trial_config, model_type, config.model.seed, target[train_index]
            )
            train_matrix = model_matrix(trial_config, model_type, features.iloc[train_index])
            valid_matrix = model_matrix(trial_config, model_type, features.iloc[valid_index])
            estimator.fit(train_matrix, target[train_index])
            oof[valid_index] = estimator.predict_proba(valid_matrix)[:, 1]
            running.append(
                float(average_precision_score(target[valid_index], oof[valid_index]))
            )
            trial.report(float(np.mean(running)), step)
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

        calibrator = fit_calibrator(trial_config, oof, target)
        calibrated = np.clip(
            calibrator.predict(oof),
            trial_config.calibration.clip_low,
            trial_config.calibration.clip_high,
        )
        selection = choose_threshold(target, calibrated, value, cost, limit=limit)
        trial.set_user_attr("pr_auc_oof", float(average_precision_score(target, calibrated)))
        trial.set_user_attr("threshold", float(selection["threshold"]))
        return float(selection["net_value_per_1000_sessions"])

    return objective


def tune_study(
    config: Settings,
    model_type: str,
    trials: int,
    timeout: int | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    """Run one study and return its best parameters and score."""
    import optuna

    from shopper_intent.features.build import read_feature_schema, read_processed
    from shopper_intent.models.train import fold_indices

    require(model_type)
    frame = read_processed(config)
    schema = read_feature_schema(config)
    train_frame = frame[frame["split"] == "train"].reset_index(drop=True)
    features = train_frame[schema["features"]]
    target = train_frame[config.features.target].to_numpy().astype(int)
    folds = fold_indices(config, features, target)

    config.optuna_storage.parent.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{config.optuna_storage.as_posix()}"
    study_name = f"{config.tuning.study_prefix}-{model_type}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(
            seed=config.tuning.sampler_seed,
            n_startup_trials=config.tuning.n_startup_trials,
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=config.tuning.n_startup_trials,
            n_warmup_steps=config.tuning.n_warmup_steps,
        ),
    )
    already = len(study.trials)
    study.optimize(
        objective_for(config, model_type, features, target, folds),
        n_trials=trials,
        timeout=timeout,
        show_progress_bar=show_progress,
    )
    completed = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    if not completed:
        raise RuntimeError(f"no completed trial for {model_type}; check the search space")
    best = study.best_trial
    return {
        "model_type": model_type,
        "study": study_name,
        "trials_run": len(study.trials) - already,
        "trials_total": len(study.trials),
        "completed": len(completed),
        "pruned": sum(1 for trial in study.trials if trial.state.name == "PRUNED"),
        "best_value_per_1000": float(best.value),
        "best_params": dict(best.params),
        "best_pr_auc_oof": best.user_attrs.get("pr_auc_oof"),
        "best_threshold": best.user_attrs.get("threshold"),
    }


def tuned_config_payload(
    config: Settings,
    results: dict[str, Any],
    command: str,
) -> str:
    """YAML text with the best parameters merged into the source config."""
    raw = yaml.safe_load(config.source_path.read_text(encoding="utf-8"))
    for model_type, payload in results["models"].items():
        raw["model"][model_type].update(payload["best_params"])
    header = [
        "# Generated by `tune`; do not edit by hand.",
        f"# Command: {command}",
        f"# Generated at: {results['finished_at']}",
        f"# Source config: {config.source_path.name}",
        "#",
        "# Out-of-fold net value per 1000 sessions at the tuned threshold:",
    ]
    for model_type, payload in results["models"].items():
        header.append(
            f"#   {model_type:<9s} {payload['best_value_per_1000']:>12.2f}  "
            f"trials {payload['completed']} completed, {payload['pruned']} pruned"
        )
    header.append("")
    body = yaml.safe_dump(raw, sort_keys=False)
    return "\n".join(header) + body


def tune(
    config: Settings,
    models: list[str] | None = None,
    trials: int | None = None,
    timeout: int | None = None,
    out: Path | None = None,
    command: str = "tune",
) -> dict[str, Any]:
    """Tune each requested model and write a config carrying the best parameters."""
    models = list(models or config.tuning.models or [config.serving.model])
    trials = int(trials or config.tuning.trials)
    timeout = config.tuning.timeout_seconds if timeout is None else timeout
    started = time.time()
    results: dict[str, Any] = {
        "started_at": datetime.now(UTC).replace(microsecond=0).isoformat(),
        "trials_requested": trials,
        "timeout_seconds": timeout,
        "models": {},
    }
    for model_type in models:
        results["models"][model_type] = tune_study(
            config, model_type, trials=trials, timeout=timeout
        )
    results["finished_at"] = datetime.now(UTC).replace(microsecond=0).isoformat()
    results["duration_seconds"] = round(time.time() - started, 1)

    target = out or (config.source_path.parent / "tuned.yaml")
    target.write_text(tuned_config_payload(config, results, command), encoding="utf-8")
    results["path"] = str(target)
    return results
