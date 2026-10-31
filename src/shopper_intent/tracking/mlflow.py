"""MLflow tracking and model registry, optional by design.

``train --track`` calls :func:`log_training_run`. The helper never raises: if
MLflow is not installed, if the tracking store cannot be reached, or if model
registration is refused by the server, the run is still trained and saved to the
local registry and the reason is returned in the result dictionary. A tracking
outage must not cost a training run.

Two artifacts are logged per model:

``estimator``   the raw sklearn ``Pipeline`` through the sklearn flavor. It
                reproduces the uncalibrated scores and nothing else.
``model``       a ``pyfunc`` model wrapping the full :class:`IntentModel`. This
                is the registered model: it carries the calibrator, the business
                threshold and the conformal policy, so loading it reproduces the
                serving decision, not just the raw score.

The registered name is ``<prefix>_<model_type>``. The champion of the run gets
the configured alias (``champion`` by default) in the MLflow model registry.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

from shopper_intent.config import Settings

try:  # MLflow stays optional: the wrapper only needs the real base when logged.
    from mlflow.pyfunc import PythonModel as _PythonModel
except Exception:  # pragma: no cover - mlflow absent
    class _PythonModel:  # type: ignore[no-redef]
        """Stand-in so the module imports without MLflow installed."""


class IntentPyfunc(_PythonModel):
    """Thin MLflow wrapper so the calibrated artifact is the registered model."""

    def __init__(self, intent: Any) -> None:
        self.intent = intent

    def predict(self, context: Any, model_input: pd.DataFrame, params: dict | None = None):
        frame = model_input if isinstance(model_input, pd.DataFrame) else pd.DataFrame(model_input)
        result = self.intent.predict(frame)
        regimes = self.intent.band(result["probability"])
        return pd.DataFrame(
            {
                "probability": result["probability"],
                "decision": result["decision"],
                "band": regimes,
                "expected_value_if_contacted": result["expected_value_if_contacted"],
                "expected_value_of_decision": result["expected_value_of_decision"],
            }
        )


def _flat_metrics(prefix: str, metrics: dict[str, Any]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[f"{prefix}/{key}"] = float(value)
    return flat


def _write_metadata(model: Any, metadata: dict[str, Any], directory: Path) -> Path:
    path = directory / f"{model.model_type}_metadata.json"
    path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return path


def _registered_version(info: Any) -> str | None:
    """Read the version whether MLflow returns a number or a ModelVersion."""
    raw = getattr(info, "registered_model_version", None)
    raw = getattr(raw, "version", raw)
    return str(raw) if raw is not None else None


def log_training_run(config: Settings, results: dict[str, Any]) -> dict[str, Any]:
    """Log one MLflow run for a trained version. Returns what happened, never raises."""
    try:
        import mlflow
        from mlflow import MlflowClient
    except Exception as error:  # noqa: BLE001 - tracking is optional
        return {"enabled": False, "reason": f"mlflow unavailable: {error}"}

    outcome: dict[str, Any] = {
        "enabled": True,
        "tracking_uri": config.tracking.tracking_uri,
        "experiment": config.tracking.experiment,
        "run_id": None,
        "models": {},
        "champion": results.get("champion"),
        "champion_alias": config.tracking.champion_alias,
        "fallback": None,
    }
    try:
        import os

        config.mlruns_dir.mkdir(parents=True, exist_ok=True)
        uri = config.tracking.tracking_uri
        if uri.startswith("file:"):
            # MLflow 3 put the filesystem backend in maintenance mode and refuses
            # to open it unless this process opts in. The run still logs; the
            # store simply cannot do everything a database can.
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
            uri = f"file:{config.mlruns_dir.as_posix()}"
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(config.tracking.experiment)
        outcome["tracking_uri"] = uri

        from shopper_intent.models import registry

        with mlflow.start_run(run_name=f"train-{results['version']}") as run:
            outcome["run_id"] = run.info.run_id
            mlflow.set_tags(
                {
                    "config_hash": config.config_hash(),
                    "config_path": str(config.source_path),
                    "version": results["version"],
                    "selection_rule": results.get("selection_rule"),
                    "candidates": ",".join(results["models"]),
                    "python": "3.12",
                }
            )
            mlflow.log_params(
                {
                    "business.value_per_conversion": config.business.value_per_conversion,
                    "business.cost_per_outreach": config.business.cost_per_outreach,
                    "model.seed": config.model.seed,
                    "model.cv_folds": config.model.cv_folds,
                    "calibration.method": config.calibration.method,
                    "conformal.alpha": config.conformal.alpha,
                    "features.test_size": config.features.test_size,
                }
            )

            for model_type, metadata in results["models"].items():
                try:
                    model, _ = registry.load(
                        config.models_dir, model_type, metadata["version"]
                    )
                except Exception as error:  # noqa: BLE001 - keep logging the rest
                    outcome["models"][model_type] = {"error": str(error)}
                    continue

                prefix = model_type
                mlflow.log_params(
                    {
                        f"{prefix}.threshold": model.threshold,
                        f"{prefix}.calibration": str(metadata.get("calibration", {}).get("method")),
                        f"{prefix}.native_categorical": model.native_categorical,
                        f"{prefix}.cv_folds": metadata["protocol"]["cv_folds"],
                        f"{prefix}.train_rows": metadata["protocol"]["train_rows"],
                        f"{prefix}.test_rows": metadata["protocol"]["test_rows"],
                    }
                )
                mlflow.log_metrics(_flat_metrics(f"{prefix}.cv", metadata["metrics"]["cv"]))
                mlflow.log_metrics(_flat_metrics(f"{prefix}.test", metadata["metrics"]["test"]))
                conformal = metadata["metrics"].get("conformal_test") or {}
                if "coverage" in conformal:
                    mlflow.log_metric(f"{prefix}.conformal.coverage", conformal["coverage"])
                    mlflow.log_metric(f"{prefix}.conformal.q_hat", conformal["q_hat"])

                with tempfile.TemporaryDirectory() as temporary:
                    directory = Path(temporary)
                    metadata_path = _write_metadata(model, metadata, directory)
                    joblib_path = directory / "model.joblib"
                    import joblib

                    joblib.dump(model, joblib_path)
                    mlflow.log_artifact(str(metadata_path))
                    mlflow.log_artifact(str(joblib_path))

                    mlflow.sklearn.log_model(
                        model.estimator,
                        name=f"{prefix}_estimator",
                        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                        registered_model_name=None,
                    )
                    registered_name = f"{config.tracking.registered_model_prefix}_{model_type}"
                    info = mlflow.pyfunc.log_model(
                        name=f"{prefix}_model",
                        python_model=IntentPyfunc(model),
                        registered_model_name=registered_name,
                        pip_requirements=["pandas", "numpy", "scikit-learn", "joblib", "pyyaml"],
                    )

                outcome["models"][model_type] = {
                    "registered_model": registered_name,
                    "version": _registered_version(info),
                    "threshold": model.threshold,
                    "calibration": metadata.get("calibration", {}).get("method"),
                }

            champion = results.get("champion")
            if champion and champion in outcome["models"]:
                entry = outcome["models"][champion]
                try:
                    client = MlflowClient()
                    client.set_registered_model_alias(
                        entry["registered_model"],
                        config.tracking.champion_alias,
                        entry["version"],
                    )
                    entry["alias"] = config.tracking.champion_alias
                    mlflow.set_tag("champion", f"{entry['registered_model']}@"
                                              f"{config.tracking.champion_alias}")
                except Exception as error:  # noqa: BLE001 - alias is a convenience
                    entry["alias_error"] = str(error)
                    outcome["fallback"] = f"alias not set: {error}"
    except Exception as error:  # noqa: BLE001 - never fail a training run for tracking
        outcome["fallback"] = f"{type(error).__name__}: {error}"
    return outcome
